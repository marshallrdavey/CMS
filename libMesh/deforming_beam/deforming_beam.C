// \author Marshall Davey
// \date August 2020
// Adapted from libMesh systems_of_equations_ex7
//
// In this example, we consider an elastic cantilever beam modeled with an
// incompressible, neohookean constitutive model. The implementation 
// presented here uses NonlinearImplicitSystem. The following equation is used to 
// define work.
// 
// W(CC) = a*(tr(CC_bar) - 3)
//
// where:
//  * u is the displacement 
//  * FF is the deformation gradient tensor, I + du/dX
//  * J is the volume change, det(FF)
//  * FF_bar is the deviatoric component of FF, J^(-1/3)FF
//  * CC is the right Cuachy-Green tensor, FF^T FF
//  * CC_bar is the deviatoric component of C, J^(-2/3)CC
// 
// We formulate the PDE on the reference geometry (\Omega) as opposed to the deformed
// geometry (\Omega^deformed). 
//
//     \int_\Omega PP_ij v_i,j = \int_\Omega f_i v_i + \int_\Gamma g_i v_i ds
//
// where:
//  * PP is the first Piola-Kirchhoff stress tensor, dW(CC)/dFF 
//  * v is the velocity
//  * f is a body load
//  * g is a surface traction on the surface \Gamma
//
// In this example we only consider a body load (e.g. gravity), hence we set g = 0. We use
// an ExplicitSystem to ouput the J value of each element given a solution. 

// C++ include files that we need
#include <iostream>
#include <algorithm>
#include <cmath>

// Various include files needed for the mesh & solver functionality.
#include "libmesh/libmesh.h"
#include "libmesh/mesh.h"
#include "libmesh/mesh_refinement.h"
#include "libmesh/exodusII_io.h"
#include "libmesh/equation_systems.h"
#include "libmesh/fe.h"
#include "libmesh/quadrature_gauss.h"
#include "libmesh/dof_map.h"
#include "libmesh/sparse_matrix.h"
#include "libmesh/numeric_vector.h"
#include "libmesh/dense_matrix.h"
#include "libmesh/dense_vector.h"
#include "libmesh/elem.h"
#include "libmesh/string_to_enum.h"
#include "libmesh/getpot.h"
#include "libmesh/mesh_generation.h"
#include "libmesh/dirichlet_boundaries.h"
#include "libmesh/zero_function.h"
#include "libmesh/enum_solver_package.h"
#include "libmesh/exact_solution.h"

// The nonlinear solver and system we will be using
#include "libmesh/nonlinear_solver.h"
#include "libmesh/nonlinear_implicit_system.h"

#define BOUNDARY_ID_MIN_Z 0
#define BOUNDARY_ID_MIN_Y 1
#define BOUNDARY_ID_MAX_X 2
#define BOUNDARY_ID_MAX_Y 3
#define BOUNDARY_ID_MIN_X 4
#define BOUNDARY_ID_MAX_Z 5

using namespace libMesh;



class LargeDeformationElasticity : public NonlinearImplicitSystem::ComputeResidual,
                                   public NonlinearImplicitSystem::ComputeJacobian
{
private:
  EquationSystems & es;

public:

  LargeDeformationElasticity (EquationSystems & es_in) :
    es(es_in)
  {}

  /**
   * Kronecker delta function.
   */
  Real delta(unsigned int i,
             unsigned int j)
  {
    return i == j ? 1. : 0.;
  }
    
  /**
   * Evaluate the fist invariant 
   */ 
  Real I_1(const TensorValue<Number> & FF)
  {
    TensorValue<Number> CC = FF.transpose()*FF;
    return CC.tr();
  }

  /**
   * PK1 stress
   */
  TensorValue<Number> PK1_stress(const TensorValue<Number> & FF,
                                 const Real & I_1,
                                 const Real & J,
                                 const Real & a)
  {
    TensorValue<Number> PP = a/cbrt(J*J)*(FF - I_1*FF.inverse().transpose()/3.0);
    return PP;
  }

  /**
   * dPP/dFF
   */
  DenseMatrix<Number> script_A(const TensorValue<Number> & FF,
                               const Real & I_1,
                               const Real & J,
                               const Real & a)
  {
    DenseMatrix<Number> A(9,9);
    TensorValue<Number> FF_inv = FF.inverse();
    for (unsigned int i = 0; i < 3; i++)
    {
      for (unsigned int j = 0; j < 3; j++)
      {
        for (unsigned int k = 0; k < 3; k++)
        {
          for (unsigned int l = 0; l < 3; l++)
          {
            A(3*i + k,3*j + l) = a/cbrt(J*J)*( 2.0*I_1*FF_inv(j,i)*FF_inv(l,k)
                                             - 2.0/3.0*FF(i,j)*FF_inv(l,k)
                                             + delta(i,k)*delta(j,l)
                                             - 2.0/3.0*FF(k,l)*FF_inv(j,i)
                                             + I_1/3.0*FF_inv(j,k)*FF_inv(l,i));
          }
        }
      }
    }            
    return A;
  } 

  /**
   * Evaluate the Jacobian of the nonlinear system.
   */
  virtual void jacobian (const NumericVector<Number> & soln,
                         SparseMatrix<Number> & jacobian,
                         NonlinearImplicitSystem & /*sys*/)
  {
    // get model parameters
    const Real a = es.parameters.get<Real>("a");
    const Real b = es.parameters.get<Real>("b"); 

    // get mesh
    const MeshBase & mesh = es.get_mesh();
    const unsigned int dim = mesh.mesh_dimension();

    // get implicit system for solve
    NonlinearImplicitSystem & system =
      es.get_system<NonlinearImplicitSystem>("NonlinearElasticity");
    
    // get u variable id
    const unsigned int u_var = system.variable_number ("u");                // wlog, since u, v, w all same

    // get system dof map
    const DofMap & dof_map = system.get_dof_map();

    // set up fe type and attach quadrature rule
    FEType fe_type = dof_map.variable_type(u_var);
    std::unique_ptr<FEBase> fe (FEBase::build(dim, fe_type));
    QGauss qrule (dim, fe_type.default_quadrature_order());
    fe->attach_quadrature_rule (&qrule);
   
    // initialize fe functions 
    const std::vector<Real> & JxW = fe->get_JxW();                          // JxW for each quadrature point, [qp] 
    const std::vector<std::vector<Real>> & phi = fe->get_phi();             // phi(qp), [node][qp]
    const std::vector<std::vector<RealGradient>> & dphi = fe->get_dphi();   // dphi(qp), [node][qp](dim)

    // initialize elemental tangent stiffness matrix
    DenseMatrix<Number> Ke;
    DenseSubMatrix<Number> Ke_var[3][3] = 
      {
        {DenseSubMatrix<Number>(Ke), DenseSubMatrix<Number>(Ke), DenseSubMatrix<Number>(Ke)},
        {DenseSubMatrix<Number>(Ke), DenseSubMatrix<Number>(Ke), DenseSubMatrix<Number>(Ke)},
        {DenseSubMatrix<Number>(Ke), DenseSubMatrix<Number>(Ke), DenseSubMatrix<Number>(Ke)}
      };

    // initialize dof inidices
    std::vector<dof_id_type> dof_indices;                                   // all dof indices, [var = 3 x node = 8] 
    std::vector<std::vector<dof_id_type>> dof_indices_var(3);               // dof indices partitioned by variable, [var = 3][node = 8]

    // clear jacobian
    jacobian.zero();

    // loop over elements
    for (const auto & elem : mesh.active_local_element_ptr_range())
    {
      // fill the dof vectors
      dof_map.dof_indices (elem, dof_indices);
      for (unsigned int var=0; var<3; var++)
      {
        dof_map.dof_indices (elem, dof_indices_var[var], var);
      }

      // grab the sizes of the dof vectors
      const unsigned int n_dofs = dof_indices.size();
      const unsigned int n_var_dofs = dof_indices_var[0].size();
        
      // initialize fe to the specific element
      fe->reinit (elem);

      // create 24x24 dense matrix
      Ke.resize (n_dofs, n_dofs);

      // fill dense matrix with 9 8x8 submatrices
      for (unsigned int var_i=0; var_i<3; var_i++)
      {
        for (unsigned int var_j=0; var_j<3; var_j++)
        {
          Ke_var[var_i][var_j].reposition (var_i*n_var_dofs, var_j*n_var_dofs, n_var_dofs, n_var_dofs);
        }
      }
        
      // loop over quadrature points        
      for (unsigned int qp=0; qp<qrule.n_points(); qp++)
      {
        DenseVector<Number> u_vec(3);
        TensorValue<Number> grad_u;
        // create u and du/dX for qp
        for (unsigned int var_i=0; var_i<3; var_i++)
        {
          // u(qp) = \sum u_j phi_j(qp)
          for (unsigned int j=0; j<n_var_dofs; j++)
          {
            u_vec(var_i) += phi[j][qp]*soln(dof_indices_var[var_i][j]);
          }

          // du(qp)/dX = \sum u_j \otimes phi_j(qp)
          for (unsigned int var_j=0; var_j<3; var_j++)
          {  
            for (unsigned int j=0; j<n_var_dofs; j++)
            {
              grad_u(var_i,var_j) += dphi[j][qp](var_j)*soln(dof_indices_var[var_i][j]);
            }
          }    
        }

        // find the deformation gradient            
        TensorValue<Number> FF = grad_u;
        // F = I + grad_u
        for (unsigned int var=0; var<3; var++)
        {
          FF(var, var) += 1.0;
        }

        // generate the first invarint using FF
        const Real I1 = I_1(FF);

        // find J
        const Number J = FF.det();

        // generate script_A for Ke
        const DenseMatrix<Number> AA = script_A(FF, I1, J, a); 
            
        // generate tangent stiffness matrix
        // to end up with Ke*([u,v,w]^T)
        for (unsigned int dof_i=0; dof_i<n_var_dofs; dof_i++)
        {
          for (unsigned int dof_j=0; dof_j<n_var_dofs; dof_j++)
          {
            for (unsigned int i=0; i<3; i++)
            {
              for (unsigned int j=0; j<3; j++)
              {
                for (unsigned int k=0; k<3; k++)
                {
                  for (unsigned int l=0; l<3; l++)
                  {
                    Ke_var[i][j](dof_i,dof_j) += AA(3*i + j, 3*k + l)*dphi[dof_i][qp](k)*dphi[dof_j][qp](l)*JxW[qp]; 
                  }
                }
              }
            }
          }
        }
      } // end qp loop
      // map Ke entries to dof indicies
      dof_map.constrain_element_matrix (Ke, dof_indices);
      // add Ke to the correct dof location in the sparse matrix jacobian
      jacobian.add_matrix (Ke, dof_indices);
    } // end element loop
  }

  /**
   * Evaluate the residual of the nonlinear system.
   */
  virtual void residual (const NumericVector<Number> & soln,
                         NumericVector<Number> & residual,
                         NonlinearImplicitSystem & /*sys*/)
  {
    const Real a = es.parameters.get<Real>("a");
    const Real b = es.parameters.get<Real>("b");
    const Real x_forcing_magnitude = es.parameters.get<Real>("x_forcing_magnitude");
    const Real y_forcing_magnitude = es.parameters.get<Real>("y_forcing_magnitude");
    const Real z_forcing_magnitude = es.parameters.get<Real>("z_forcing_magnitude");

    const MeshBase & mesh = es.get_mesh();
    const unsigned int dim = mesh.mesh_dimension();

    NonlinearImplicitSystem & system =
      es.get_system<NonlinearImplicitSystem>("NonlinearElasticity");

    const unsigned int u_var = system.variable_number ("u");

    const DofMap & dof_map = system.get_dof_map();

    FEType fe_type = dof_map.variable_type(u_var);
    std::unique_ptr<FEBase> fe (FEBase::build(dim, fe_type));
    QGauss qrule (dim, fe_type.default_quadrature_order());
    fe->attach_quadrature_rule (&qrule);

    const std::vector<Real> & JxW = fe->get_JxW();
    const std::vector<std::vector<Real>> & phi = fe->get_phi();
    const std::vector<std::vector<RealGradient>> & dphi = fe->get_dphi();

    // create residual dense vector
    DenseVector<Number> Re;
    
    // create residual subvectors
    DenseSubVector<Number> Re_var[3] =
      {DenseSubVector<Number>(Re),
       DenseSubVector<Number>(Re),
       DenseSubVector<Number>(Re)};

    std::vector<dof_id_type> dof_indices;
    std::vector<std::vector<dof_id_type>> dof_indices_var(3);

    residual.zero();

    // loop over elements
    for (const auto & elem : mesh.active_local_element_ptr_range())
    { 
      // create dof maps
      dof_map.dof_indices (elem, dof_indices);
      for (unsigned int var=0; var<3; var++)
      {
        dof_map.dof_indices (elem, dof_indices_var[var], var);
      }

      const unsigned int n_dofs = dof_indices.size();
      const unsigned int n_var_dofs = dof_indices_var[0].size();

      fe->reinit (elem);

      Re.resize (n_dofs);
      for (unsigned int var=0; var<3; var++)
      {
        Re_var[var].reposition (var*n_var_dofs, n_var_dofs);
      }
      
      // loop over quadrature points
      for (unsigned int qp=0; qp<qrule.n_points(); qp++)
      {
        // u and du/dX
        DenseVector<Number> u_vec(3);
        TensorValue<Number> grad_u;
        for (unsigned int var_i=0; var_i<3; var_i++)
        {
          for (unsigned int j=0; j<n_var_dofs; j++)
          {
            u_vec(var_i) += phi[j][qp]*soln(dof_indices_var[var_i][j]);
          }
          
          // Row is variable u, v, or w column is x, y, or z
          for (unsigned int var_j=0; var_j<3; var_j++)
          {
            for (unsigned int j=0; j<n_var_dofs; j++)
            {            
              grad_u(var_i,var_j) += dphi[j][qp](var_j)*soln(dof_indices_var[var_i][j]);
            }
          }
        }

        // Define the deformation gradient
        TensorValue<Number> FF = grad_u;
        for (unsigned int var=0; var<3; var++)
        {
          FF(var, var) += 1.0;
        }
        
        // generate I_1 from FF
        const Real I1 = I_1(FF);

        // generat J
        const Number J = FF.det();
        
        // generate PK1
        const TensorValue<Number> PP = PK1_stress(FF, I1, J, a);

        // construct force vector
        DenseVector<Number> f_vec(3);
        f_vec(0) = x_forcing_magnitude;
        f_vec(1) = y_forcing_magnitude;
        f_vec(2) = z_forcing_magnitude;

        // construct rhs vector
        for (unsigned int dof_i=0; dof_i<n_var_dofs; dof_i++)
        {
          for (unsigned int i=0; i<3; i++)
          {
            // \int f*phi_i(qp) - PP(qp)dphi_i(qp)
            for (unsigned int j=0; j<3; j++)
            {
              Re_var[i](dof_i) += -PP(i,j)*dphi[dof_i][qp](j)*JxW[qp];
            }
            // \int phi_i(qp)*f(i)  
            Re_var[i](dof_i) += f_vec(i)*phi[dof_i][qp]*JxW[qp];
          }
        }
      } // end qp loop
      // constrains RHS vector
      dof_map.constrain_element_vector (Re, dof_indices);
      // add vector to residual spare vector
      residual.add_vector (Re, dof_indices);
    } // end element loop
  }

  /**
   * Compute the average J value for each element at the current solution.
   */
  void compute_J()
  {
    const MeshBase & mesh = es.get_mesh();
    const unsigned int dim = mesh.mesh_dimension();
    
    NonlinearImplicitSystem & system =
      es.get_system<NonlinearImplicitSystem>("NonlinearElasticity");
    
    unsigned int displacement_vars[3];
    displacement_vars[0] = system.variable_number ("u");
    displacement_vars[1] = system.variable_number ("v");
    displacement_vars[2] = system.variable_number ("w");
    const unsigned int u_var = system.variable_number ("u");

    const DofMap & dof_map = system.get_dof_map();
    FEType fe_type = dof_map.variable_type(u_var);
    std::unique_ptr<FEBase> fe (FEBase::build(dim, fe_type));
    QGauss qrule (dim, fe_type.default_quadrature_order());
    fe->attach_quadrature_rule (&qrule);

    const std::vector<Real> & JxW = fe->get_JxW();
    const std::vector<std::vector<RealGradient>> & dphi = fe->get_dphi();

    // Also, get a reference to the ExplicitSystem
    ExplicitSystem & J_system = es.get_system<ExplicitSystem>("JSystem");
    const DofMap & J_dof_map = J_system.get_dof_map();

    unsigned int J_var = J_system.variable_number ("J");

    // Storage for the stress dof indices on each element
    std::vector<std::vector<dof_id_type>> dof_indices_var(system.n_vars());
    std::vector<dof_id_type> J_dof_indices_var;
    
    // new element volume
    Number current_volume;

    // element loop
    for (const auto & elem : mesh.active_local_element_ptr_range())
    {
      current_volume = 0.0;

      for (unsigned int var=0; var<3; var++)
      {
          dof_map.dof_indices (elem, dof_indices_var[var], displacement_vars[var]);
      }

      const unsigned int n_var_dofs = dof_indices_var[0].size();

      fe->reinit (elem);

      for (unsigned int qp=0; qp<qrule.n_points(); qp++)
      {
        TensorValue<Number> grad_u;
        for (unsigned int var_i=0; var_i<3; var_i++)
        {
          for (unsigned int var_j=0; var_j<3; var_j++)
          {
            for (unsigned int j=0; j<n_var_dofs; j++)
            {
              grad_u(var_i,var_j) += dphi[j][qp](var_j) * system.current_solution(dof_indices_var[var_i][j]);
            }
          }
        }

        TensorValue<Number> FF = grad_u;
        for (unsigned int var=0; var<3; var++)
        {
          FF(var, var) += 1.0;
        }

        current_volume += FF.det()*JxW[qp];
      }// end qp loop

      const Number J = current_volume/elem->volume();

      J_dof_map.dof_indices (elem, J_dof_indices_var, J_var);
      
      // constant monomial, so only one dof index
      dof_id_type dof_index = J_dof_indices_var[0];

      J_system.solution->set(dof_index, J);
    }// end element loop

    J_system.solution->close();
    J_system.update();
  }
};

int main (int argc, char ** argv)
{
  LibMeshInit init (argc, argv);

  // This example requires the PETSc nonlinear solvers
  libmesh_example_requires(libMesh::default_solver_package() == PETSC_SOLVERS, "--enable-petsc");

  // We use a 3D domain.
  libmesh_example_requires(LIBMESH_DIM > 2, "--disable-1D-only --disable-2D-only");

  GetPot infile(argv[1]);
  const Real x_length = infile("x_length", 0.);
  const Real y_length = infile("y_length", 0.);
  const Real z_length = infile("z_length", 0.);
  const Real n_elem_x = infile("n_elem_x", 0);
  const Real n_elem_y = infile("n_elem_y", 0);
  const Real n_elem_z = infile("n_elem_z", 0);
  const std::string approx_order = infile("approx_order", "FIRST");
  const std::string fe_family = infile("fe_family", "LAGRANGE");

  const Real a = infile("a", 100.0);
  const Real b = infile("b", 1.0);
  const Real x_forcing_magnitude = infile("x_forcing_magnitude", 0.0);
  const Real y_forcing_magnitude = infile("y_forcing_magnitude", 0.0);
  const Real z_forcing_magnitude = infile("z_forcing_magnitude", 0.0);

  const Real nonlinear_abs_tol = infile("nonlinear_abs_tol", 1.e-8);
  const Real nonlinear_rel_tol = infile("nonlinear_rel_tol", 1.e-8);
  const unsigned int nonlinear_max_its = infile("nonlinear_max_its", 50);

  const unsigned int n_solves = infile("n_solves", 10);
  const Real force_scaling = infile("force_scaling", 5.0);

  Mesh mesh(init.comm());

  MeshTools::Generation::build_cube(mesh,
                                    n_elem_x,
                                    n_elem_y,
                                    n_elem_z,
                                    0., x_length,
                                    0., y_length,
                                    0., z_length,
                                    HEX27);

  mesh.print_info();

  EquationSystems equation_systems (mesh);
  LargeDeformationElasticity lde(equation_systems);

  NonlinearImplicitSystem & system =
    equation_systems.add_system<NonlinearImplicitSystem> ("NonlinearElasticity");

  unsigned int u_var =
    system.add_variable("u",
                        Utility::string_to_enum<Order>   (approx_order),
                        Utility::string_to_enum<FEFamily>(fe_family));

  unsigned int v_var =
    system.add_variable("v",
                        Utility::string_to_enum<Order>   (approx_order),
                        Utility::string_to_enum<FEFamily>(fe_family));

  unsigned int w_var =
    system.add_variable("w",
                        Utility::string_to_enum<Order>   (approx_order),
                        Utility::string_to_enum<FEFamily>(fe_family));

  ExplicitSystem & J_system = 
    equation_systems.add_system<ExplicitSystem> ("JSystem");
  
  J_system.add_variable("J", CONSTANT, MONOMIAL);
 
  equation_systems.parameters.set<Real>         ("nonlinear solver absolute residual tolerance") = nonlinear_abs_tol;
  equation_systems.parameters.set<Real>         ("nonlinear solver relative residual tolerance") = nonlinear_rel_tol;
  equation_systems.parameters.set<unsigned int> ("nonlinear solver maximum iterations")          = nonlinear_max_its;

  system.nonlinear_solver->residual_object = &lde;
  system.nonlinear_solver->jacobian_object = &lde;

  equation_systems.parameters.set<Real>("a") = a;
  equation_systems.parameters.set<Real>("b") = b;
  equation_systems.parameters.set<Real>("x_forcing_magnitude") = x_forcing_magnitude;
  equation_systems.parameters.set<Real>("y_forcing_magnitude") = y_forcing_magnitude;
  equation_systems.parameters.set<Real>("z_forcing_magnitude") = z_forcing_magnitude;

  // Attach Dirichlet boundary conditions
  std::set<boundary_id_type> clamped_boundaries;
  clamped_boundaries.insert(BOUNDARY_ID_MIN_X);

  std::vector<unsigned int> uvw;
  uvw.push_back(u_var);
  uvw.push_back(v_var);
  uvw.push_back(w_var);

  ZeroFunction<Number> zero;

  // Most DirichletBoundary users will want to supply a "locally
  // indexed" functor
  system.get_dof_map().add_dirichlet_boundary
    (DirichletBoundary (clamped_boundaries, uvw, zero,
                        LOCAL_VARIABLE_ORDER));

  equation_systems.init();
  equation_systems.print_info();

  // Provide a loop here so that we can do a sequence of solves
  // where solve n gives a good starting guess for solve n+1.
  // This "continuation" approach is helpful for solving for
  // large values of "forcing_magnitude".
  // Set n_solves and force_scaling in nonlinear_elasticity.in.
  for (unsigned int count=0; count<n_solves; count++)
    {
      equation_systems.parameters.set<Real>("x_forcing_magnitude") = 
        x_forcing_magnitude*pow(force_scaling, count);

      equation_systems.parameters.set<Real>("y_forcing_magnitude") = 
        y_forcing_magnitude*pow(force_scaling, count);

      equation_systems.parameters.set<Real>("z_forcing_magnitude") = 
        z_forcing_magnitude*pow(force_scaling, count);

      libMesh::out << "Performing solve "
                   << count
                   << ", forcing_vector: ("
                   << equation_systems.parameters.get<Real>("x_forcing_magnitude")
                   << ", "
                   << equation_systems.parameters.get<Real>("y_forcing_magnitude")
                   << ", "
                   << equation_systems.parameters.get<Real>("z_forcing_magnitude")
                   << ")\n";

      system.solve();

      libMesh::out << "System solved at nonlinear iteration "
                   << system.n_nonlinear_iterations()
                   << " , final nonlinear residual norm: "
                   << system.final_nonlinear_residual()
                   << std::endl;

      libMesh::out << "Computing J values...\n\n";

      lde.compute_J();

#ifdef LIBMESH_HAVE_EXODUS_API
      std::stringstream filename;
      filename << "solution_" << count << ".exo";
      ExodusII_IO (mesh).write_equation_systems(filename.str(), equation_systems);
#endif
    }

  return 0;
}
