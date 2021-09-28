#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/shared_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/constrained_linear_operator.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/solver_gmres.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/physics/elasticity/kinematics.h>

#include <fstream>
#include <math.h>

// This is a program to deform a beam with body force in the absence of momentum.
// A 2-dimensional implementation is not supported in this model.

using namespace dealii;

// define PETSc namespace for brevity
namespace la
{
    using namespace LinearAlgebraPETSc;
}

// Generate beam
template<int dim>
void make_triangulation(parallel::shared::Triangulation<dim>& triangulation,
                        const int n_global_refinements)
{
    const Point<3> c0(0, 0, 0);
    const Point<3> c1(1, 1, 2);
    const std::vector<unsigned int> rep{1, 1, 2};  
    GridGenerator::subdivided_hyper_rectangle(triangulation,
                                              rep,          // number of cells in each direction
                                              c0,           // origin corner
                                              c1,           // opposite corner
                                              true);        // colorize   
    triangulation.refine_global(n_global_refinements);
    return;
}

// class declaration
template <int dim>
class implicit_beam
{
public:
    implicit_beam(const parallel::shared::Triangulation<dim>& triangulation,
                  const unsigned int fe_order,
                  const double alpha);

    void run();

private:
    // class functions
    void setup_system();
    void assemble_mass_matrix();
    void update_force();
    void initialize_acceleration();
    void intermediate_step();
    void update_step();
    void assemble_system();
    void solve();
    void output_results(const unsigned int& step);

    // MPI communicator
    MPI_Comm mpi_comm;
    ConditionalOStream pcout;

    // mesh and finite elementparameters
    SmartPointer<const parallel::shared::Triangulation<dim>> tria;
    bool use_simplex;
    unsigned int fe_order;
    std::unique_ptr<FESystem<dim>> fe;
    std::unique_ptr<Quadrature<dim>> quadrature_formula;
    std::unique_ptr<Quadrature<dim-1>> quadrature_formula_face; 
    DoFHandler<dim> dof_handler;
    IndexSet locally_owned_dofs;
    IndexSet locally_relevant_dofs;
    
    // functions for mechanics
    double alpha;
    const double kappa = 1.;
    const double rho = 1.;
    const double eta_damping = 0.0003;
    double delta(const unsigned int i, 
                 const unsigned int j);
    double I_1(const Tensor<2, dim>& FF);
    Tensor<2, dim> PK1_stress(const Tensor<2, dim>& FF,
                              const double& I1,
                              const double& J);
    Tensor<4, dim> script_A(const Tensor<2, dim>& FF,
                            const double& I1,
                            const double& J);

    // body force and pressure
    const Vector<double> body_force{0., 0.001, 0.};
    const double pressure = 0.;
    const Vector<double> pressure_vector{0., 0., 0.};

    // numerical parameters
    const double beta = 0.25;
    const double gamma = 0.5;

    // time
    const double dt = 0.1;
    const double end_time = 80.;
    double time;

    // constraints
    AffineConstraints<double> constraints;

    // matrix assembly
    la::MPI::SparseMatrix constrained_mass_matrix;
    la::MPI::SparseMatrix unconstrained_mass_matrix;
    la::MPI::SparseMatrix constrained_system_matrix;
    la::MPI::SparseMatrix unconstrained_system_matrix;

    // solution vectors
    la::MPI::Vector displacement;         // total displacement
    la::MPI::Vector local_displacement;   
    la::MPI::Vector displacement_tilde;   // intermediate displacement
    la::MPI::Vector velocity;             // current velocity
    la::MPI::Vector local_velocity; 
    la::MPI::Vector velocity_tilde;       // intermediate velocity
    la::MPI::Vector acceleration;         // current acceleration
    la::MPI::Vector local_acceleration;   
    la::MPI::Vector force;                // current internal and external force
    la::MPI::Vector local_force;          
    la::MPI::Vector newton_update;        // displacement update from the newton iteration
    la::MPI::Vector residual;             // residual
    la::MPI::Vector local_residual;       
    la::MPI::Vector constrained_residual; // constrained residual, system rhs
    const double rtol = 1.e-8;            // residual norm tolerance for Newton's method

    // J storage
    la::MPI::Vector J_vector;
};

// class constructor
template <int dim>
implicit_beam<dim>::implicit_beam(const parallel::shared::Triangulation<dim>& triangulation,
                                  const unsigned int fe_order,
                                  const double alpha)
    :mpi_comm(triangulation.get_communicator()),
     pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_comm) == 0)),
     tria(&triangulation),
     use_simplex(!triangulation.all_reference_cells_are_hyper_cube()),
     fe_order(fe_order),
     dof_handler(triangulation),
     alpha(alpha)
{
    if(use_simplex)
    {
        fe = std::make_unique<FESystem<dim>>(FE_SimplexP<dim>(fe_order), dim);
        quadrature_formula = std::make_unique<QGaussSimplex<dim>>(fe_order + 1);
        quadrature_formula_face = std::make_unique<QGaussSimplex<dim-1>>(fe_order + 1); 
    }
    else
    {
        fe = std::make_unique<FESystem<dim>>(FE_Q<dim>(fe_order), dim);
        quadrature_formula = std::make_unique<QGauss<dim>>(fe_order + 1);
        quadrature_formula_face = std::make_unique<QGauss<dim-1>>(fe_order + 1); 
    }
}

// Kronecker delta function
template <int dim>
double implicit_beam<dim>::delta(unsigned int i,
                                 unsigned int j)
{
    return i == j ? 1.: 0.;
}

// I_1 claculator, or tr(CC)
template <int dim>
double implicit_beam<dim>::I_1(const Tensor<2, dim>& FF)
{
    return trace(Physics::Elasticity::Kinematics::C(FF));
}

// PK1_stress from the deformation gradient, dW/dFF
template <int dim>
Tensor<2, dim> implicit_beam<dim>::PK1_stress(const Tensor<2, dim>& FF,
                                              const double& I1,
                                              const double& J)
{
    return alpha/cbrt(J*J)*(FF - I1*transpose(invert(FF))/3.);
}

// script_A from the deformation gradient, d^2W/dFF^2
template <int dim>
Tensor<4, dim> implicit_beam<dim>::script_A(const Tensor<2, dim>& FF,
                                            const double& I1,
                                            const double& J)
{
    Tensor<4, dim> A;
    const Tensor<2, dim> FF_inv = invert(FF);
    for(unsigned int i = 0; i < dim; ++i)
    {
        for(unsigned int j = 0; j < dim; ++j)
        {
            for(unsigned int k = 0; k < dim; ++k)
            {
                for(unsigned int l = 0; l < dim; ++l)
                {
                    A[i][j][k][l] = alpha/cbrt(J*J)*(2./9.*I1*FF_inv[j][i]*FF_inv[l][k] -
                                                     2./3.*FF[i][j]*FF_inv[l][k] +
                                                     delta(i,k)*delta(j,l) -
                                                     2./3.*FF[k][l]*FF_inv[j][i] +
                                                     I1/3.*FF_inv[j][k]*FF_inv[l][i]);
                }
            }
        }
    }
    return A;
}

// dof handler stuff
template <int dim>
void implicit_beam<dim>::setup_system()
{
    // DoF distribution
    dof_handler.distribute_dofs(*fe);
    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);    

    // solution vectors
    displacement.reinit(locally_owned_dofs, 
                        locally_relevant_dofs,
                        mpi_comm);
    local_displacement.reinit(locally_owned_dofs, mpi_comm);
    displacement_tilde.reinit(locally_owned_dofs, mpi_comm);

    velocity.reinit(locally_owned_dofs, 
                    locally_relevant_dofs,
                    mpi_comm);
    local_velocity.reinit(locally_owned_dofs, mpi_comm);
    velocity_tilde.reinit(locally_owned_dofs, mpi_comm);

    acceleration.reinit(locally_owned_dofs, 
                        locally_relevant_dofs,
                        mpi_comm);
    local_acceleration.reinit(locally_owned_dofs, mpi_comm);

    force.reinit(locally_owned_dofs, 
                 locally_relevant_dofs,
                 mpi_comm);
    local_force.reinit(locally_owned_dofs, mpi_comm);

    newton_update.reinit(locally_owned_dofs, mpi_comm);

    residual.reinit(locally_owned_dofs, 
                    locally_relevant_dofs,
                    mpi_comm);
    local_residual.reinit(locally_owned_dofs, mpi_comm);
    constrained_residual.reinit(locally_owned_dofs, mpi_comm);

    // J storage
    J_vector.reinit(mpi_comm,
                    tria->n_global_active_cells(), 
                    tria->n_locally_owned_active_cells());

    // constrain lower z boundary to 0 displacement
    constraints.clear();
    constraints.reinit(locally_relevant_dofs);
    VectorTools::interpolate_boundary_values(dof_handler,
                                             4,
                                             Functions::ZeroFunction<dim>(dim),
                                             constraints);
    constraints.close();

    // constrained matrix generation
    DynamicSparsityPattern constrained_dsp(locally_relevant_dofs);
    DoFTools::make_sparsity_pattern(dof_handler,
                                    constrained_dsp,
                                    constraints,
                                    false); //  keep constrained dofs
    SparsityTools::distribute_sparsity_pattern(constrained_dsp,
                                               locally_owned_dofs,
                                               mpi_comm,
                                               locally_relevant_dofs);
    constrained_mass_matrix.reinit(locally_owned_dofs,
                                   locally_owned_dofs,
                                   constrained_dsp,
                                   mpi_comm);
    constrained_system_matrix.reinit(locally_owned_dofs,
                                     locally_owned_dofs,
                                     constrained_dsp,
                                     mpi_comm);

    // unconstrained mass matrix
    DynamicSparsityPattern unconstrained_dsp(locally_relevant_dofs);
    DoFTools::make_sparsity_pattern(dof_handler,
                                    unconstrained_dsp);
    SparsityTools::distribute_sparsity_pattern(unconstrained_dsp,
                                               locally_owned_dofs,
                                               mpi_comm,
                                               locally_relevant_dofs);
    unconstrained_mass_matrix.reinit(locally_owned_dofs,
                                     locally_owned_dofs,
                                     unconstrained_dsp,
                                     mpi_comm);
    unconstrained_system_matrix.reinit(locally_owned_dofs,
                                       locally_owned_dofs,
                                       unconstrained_dsp,
                                       mpi_comm);
}

// assemble the mass matrix, does not change over time
template<int dim>
void implicit_beam<dim>::assemble_mass_matrix()
{
    constrained_mass_matrix = 0;
    unconstrained_mass_matrix = 0;

    // set up FEValues.
    FEValues<dim> fe_values(*fe,
                            *quadrature_formula,
                            update_values |  
                            update_JxW_values);

    // grab the dofs per cell, dim*nodes
    const unsigned int dofs_per_cell = fe->n_dofs_per_cell();

    // initialize local matrix
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);

    // dof vector for global indices
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // loop over cells
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        if(cell->is_locally_owned())
        {
            cell_matrix = 0;

            fe_values.reinit(cell);

            // loop over quadrature points
            for (const unsigned int q_index: fe_values.quadrature_point_indices())
            {
                // assemble dim-dimensional mass matrix
                // loop over dof indices
                for(const unsigned int i: fe_values.dof_indices())
                {
                    const unsigned int i_component = fe->system_to_component_index(i).first; 
                    // loop over dof indices
                    for(const unsigned int j: fe_values.dof_indices())
                    {
                        const unsigned int j_component = fe->system_to_component_index(j).first;
                        cell_matrix(i, j) += rho*((j_component == i_component)?
                                                  fe_values.shape_value(i, q_index)*
                                                  fe_values.shape_value(j, q_index)*
                                                  fe_values.JxW(q_index):
                                                  0.0);
                    }
                }
            }
            cell->get_dof_indices(local_dof_indices);
            constraints.distribute_local_to_global(
                cell_matrix, local_dof_indices, constrained_mass_matrix);
            unconstrained_mass_matrix.add(local_dof_indices, cell_matrix);
        }
    }
    constrained_mass_matrix.compress(VectorOperation::add);
    unconstrained_mass_matrix.compress(VectorOperation::add);
}

// update the force vector
template<int dim>
void implicit_beam<dim>::update_force()
{
    local_force = 0;
    J_vector = 0;

    // set up FEValues.
    FEValues<dim> fe_values(*fe,
                            *quadrature_formula,
                            update_values | update_gradients | 
                            update_JxW_values);

    // set up FEFaceValues
    FEFaceValues<dim> fe_face_values(*fe,
                                     *quadrature_formula_face,
                                     update_values | update_normal_vectors | 
                                     update_JxW_values);

    // extractor for the displacement values
    FEValuesExtractors::Vector  u_fe(0);
    std::vector<Tensor<2, dim>> qp_Grad_u;     

    // grab the dofs per cell, dim*nodes
    const unsigned int dofs_per_cell = fe->n_dofs_per_cell();

    // initialize local rhs
    Vector<double> cell_rhs(dofs_per_cell);

    // dof vector for global indices
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // J value stuff
    double new_volume;

    // loop over cells
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        if(cell->is_locally_owned())
        {
            cell_rhs   = 0;
            new_volume = 0;

            fe_values.reinit(cell);

            const unsigned int n_q_points = fe_values.get_quadrature().size();
            qp_Grad_u.resize(n_q_points);
            fe_values[u_fe].get_function_gradients(displacement, qp_Grad_u);

            // loop over quadrature points
            for (const unsigned int q_index: fe_values.quadrature_point_indices())
            {
                // deformation gradient, FF
                const Tensor<2, dim> FF = Physics::Elasticity::Kinematics::F(qp_Grad_u[q_index]);
                // first invariant, I1
                const double I1 = I_1(FF);
                // volume change, J 
                const double J = determinant(FF);
                new_volume += J*fe_values.JxW(q_index);
                // PK1 stress, PP            
                const Tensor<2, dim> PP = PK1_stress(FF, I1, J);

                // loop over dof indices
                for(const unsigned int i: fe_values.dof_indices())
                {
                    const unsigned int i_component = fe->system_to_component_index(i).first; 
                    // internal force    
                    for(unsigned int k = 0; k < dim; ++k)
                    {
                        cell_rhs(i) -= PP[i_component][k]*
                                       fe_values.shape_grad(i, q_index)[k]*
                                       fe_values.JxW(q_index);  
                    }

                    cell_rhs(i) += rho*body_force[i_component]*
                                       fe_values.shape_value(i, q_index)*
                                       fe_values.JxW(q_index);
                } 
            }   
       
            // traction on cell
            for(const auto &face : cell->face_iterators())
            {   
                // check if face is on the correct boundary
                if(face->at_boundary() && face->boundary_id() == 5)
                {
                    // initialize face
                    fe_face_values.reinit(cell, face);
    
                    for(const unsigned int f_q_index : fe_face_values.quadrature_point_indices())
                    {
                        for(const unsigned int i : fe_values.dof_indices())
                        {
                            const unsigned int i_component = fe->system_to_component_index(i).first;
                            cell_rhs(i) += pressure*
                                           pressure_vector[i_component]*
                                           fe_face_values.shape_value(i, f_q_index)*
                                           fe_face_values.JxW(f_q_index); 
                        }                     
                    }    
                
                }
            } 
   
            cell->get_dof_indices(local_dof_indices);
            local_force.add(local_dof_indices, cell_rhs);
            J_vector[cell->active_cell_index()] = new_volume/cell->measure();
        }
    }
    local_force.compress(VectorOperation::add);
    J_vector.compress(VectorOperation::insert);
    force = local_force;
}

// initialize acceleration, M a = f
template<int dim>
void implicit_beam<dim>::initialize_acceleration()
{
    // sovler settings 
    SolverControl solver_control(1000, 1e-12);
    la::SolverCG  solver(solver_control, mpi_comm);

    // preconditioner settings
    la::MPI::PreconditionAMG preconditioner;
    la::MPI::PreconditionAMG::AdditionalData data;
    data.symmetric_operator = true;
    preconditioner.initialize(constrained_mass_matrix, data);
   
    // constrain the force
    auto force_system_operator = 
        linear_operator<la::MPI::Vector>(unconstrained_mass_matrix);
    auto setup_constrained_force = constrained_right_hand_side<la::MPI::Vector>(
        constraints, force_system_operator, force);
    la::MPI::Vector force_rhs(locally_owned_dofs, mpi_comm);
    setup_constrained_force.apply(force_rhs);

    solver.solve(constrained_mass_matrix, local_acceleration, force_rhs, preconditioner);

    constraints.distribute(local_acceleration);
}

// compute velocity_tilde and displacement_tilde
template<int dim>
void implicit_beam<dim>::intermediate_step()
{
    // intermediate displacment
    displacement_tilde = 0;
    displacement_tilde.add(1.0, local_displacement);
    displacement_tilde.add(dt, local_velocity);
    displacement_tilde.add(dt*dt*(1.0-2.0*beta)/2.0, local_acceleration);

    // intermediate velocity
    velocity_tilde = 0;
    velocity_tilde.add(1.0, local_velocity);
    velocity_tilde.add(dt*(1.0-gamma), local_acceleration);
}

// update velocity, acceleration, and the residual
template<int dim>
void implicit_beam<dim>::update_step()
{
    // update acceleration
    local_acceleration = 0;
    local_acceleration.add(1.0/beta/dt/dt, local_displacement);
    local_acceleration.add(-1.0/beta/dt/dt, displacement_tilde);
    
    // update velocity
    local_velocity = 0;
    local_velocity.add(1.0, velocity_tilde);
    local_velocity.add(gamma*dt, local_acceleration);

    // update the residual
    local_residual = 0;
    local_residual.add(1.0, local_force);
    const la::MPI::Vector negative_acceleration(-1.0*local_acceleration);
    unconstrained_mass_matrix.vmult_add(local_residual, negative_acceleration);
    local_residual.add(-0.5*eta_damping, velocity_tilde);
    local_residual.add(-0.5*eta_damping, local_velocity);    

    // constrain residual
    constrained_residual = 0;
    residual = local_residual;
    auto residual_system_operator = 
        linear_operator<la::MPI::Vector>(unconstrained_system_matrix);
    auto setup_constrained_residual = constrained_right_hand_side
        <la::MPI::Vector>(constraints, residual_system_operator, residual);
    setup_constrained_residual.apply(constrained_residual);
}

// matrix assembly
template <int dim>
void implicit_beam<dim>::assemble_system()
{
    constrained_system_matrix = 0;
    unconstrained_system_matrix = 0;

    // set up FEValues.
    FEValues<dim> fe_values(*fe,
                            *quadrature_formula,
                            update_values | update_gradients | 
                            update_JxW_values);

    // extractor for the displacement values
    FEValuesExtractors::Vector  u_fe(0);
    std::vector<Tensor<2, dim>> qp_Grad_u;     

    // grab the dofs per cell, dim*nodes
    const unsigned int dofs_per_cell = fe->n_dofs_per_cell();

    // initialize local matrix and rhs
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);

    // dof vector for global indices
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
 
    // loop over cells
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        if(cell->is_locally_owned())
        {
            cell_matrix = 0;

            fe_values.reinit(cell);
        
            const unsigned int n_q_points = fe_values.get_quadrature().size();
            qp_Grad_u.resize(n_q_points);
            fe_values[u_fe].get_function_gradients(displacement, qp_Grad_u);

            // loop over quadrature points
            for (const unsigned int q_index: fe_values.quadrature_point_indices())
            {
                // deformation gradient, FF
                const Tensor<2, dim> FF = Physics::Elasticity::Kinematics::F(qp_Grad_u[q_index]);
                // first invariant, I1
                const double I1 = I_1(FF);
                // volume change, J 
                const double J = determinant(FF);
                // d^2W/dFF^2      
                const Tensor<4, dim> AA = script_A(FF, I1, J);

                // loop over dof indices
                for(const unsigned int i: fe_values.dof_indices())
                {
                    // grad u dimension of dof i
                    const unsigned int i_component = fe->system_to_component_index(i).first;
                    // loop over dof indices
                    for(const unsigned int j: fe_values.dof_indices())
                    {
                        // grad u dimension of dof j
                        const unsigned int j_component = fe->system_to_component_index(j).first;

                        // mass matrix constribution
                        cell_matrix(i, j) += rho/beta/dt/dt*
                                             ((j_component == i_component)?
                                              fe_values.shape_value(i, q_index)*
                                              fe_values.shape_value(j, q_index)*
                                              fe_values.JxW(q_index):
                                              0.0);
                        // loop over dimensions for tangent stiffness matrix
                        for(unsigned int k = 0; k < dim; ++k)
                        {
                            for(unsigned int l = 0; l < dim; ++l)
                            {
                                cell_matrix(i, j) += AA[i_component][k][j_component][l]*
                                                     fe_values.shape_grad(i, q_index)[k]*
                                                     fe_values.shape_grad(j, q_index)[l]*
                                                     fe_values.JxW(q_index);
                            }

                        }
                    }
                }
            }
            cell->get_dof_indices(local_dof_indices);
            constraints.distribute_local_to_global(
                cell_matrix, local_dof_indices, constrained_system_matrix);
            unconstrained_system_matrix.add(local_dof_indices, cell_matrix);
        }
    }
    constrained_system_matrix.compress(VectorOperation::add);
    unconstrained_system_matrix.compress(VectorOperation::add);
}

// matrix solve
template <int dim>
void implicit_beam<dim>::solve()
{
    // intialize solver
    SolverControl   solver_control(1000, 1e-12);
    la::SolverGMRES solver(solver_control, mpi_comm);

    // preconditioner settings
    la::MPI::PreconditionJacobi preconditioner;
    la::MPI::PreconditionJacobi::AdditionalData data;
    preconditioner.initialize(constrained_system_matrix, data);
   
    // solve for newton update 
    newton_update = 0;
    solver.solve(constrained_system_matrix, newton_update, constrained_residual, preconditioner);

    // constrain newton update
    constraints.distribute(newton_update);
}

// ouptut function
template <int dim>
void implicit_beam<dim>::output_results(const unsigned int& step)
{
    // build data out object
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);

    // define displacement names
    std::vector<std::string> displacement_names;
    displacement_names.emplace_back("x_displacement");
    displacement_names.emplace_back("y_displacement");
    displacement_names.emplace_back("z_displacement");

    // define velocity names
    std::vector<std::string> velocity_names;
    velocity_names.emplace_back("x_velocity");
    velocity_names.emplace_back("y_velocity");
    velocity_names.emplace_back("z_velocity");

    // define acceleration names
    std::vector<std::string> acceleration_names;
    acceleration_names.emplace_back("x_acceleration");
    acceleration_names.emplace_back("y_acceleration");
    acceleration_names.emplace_back("z_acceleration");
    
    // update velocity and acceleration to ghosted vectors
    velocity = local_velocity;
    acceleration = local_acceleration;

    // add data to data out object
    data_out.add_data_vector(displacement, displacement_names);
    data_out.add_data_vector(velocity, velocity_names);
    data_out.add_data_vector(acceleration, acceleration_names);
    data_out.add_data_vector(J_vector, "J");

    // correlate time to time step
    data_out.set_flags(DataOutBase::VtkFlags(time, step));

    // build patches and write in parallel
    data_out.build_patches();
    if(use_simplex)
        data_out.write_vtu_with_pvtu_record("./simplex_output/", "solution", step, mpi_comm, 4);
    else
        data_out.write_vtu_with_pvtu_record("./output/", "solution", step, mpi_comm, 4);
}

// run function
template <int dim>
void implicit_beam<dim>::run()
{
    pcout << " Number of active cells:       "
          << tria->n_active_cells() << std::endl;
    setup_system();
    pcout << " Number of degrees of freedom: " << dof_handler.n_dofs()
          << std::endl;

    pcout << " Maximal cell diameter: " << GridTools::maximal_cell_diameter(*tria)
          << "\n\n";

    assemble_mass_matrix();

    // initialize problem
    time = 0.;
    unsigned int step = 0;
    unsigned int out_step = 0;
    update_force();
    initialize_acceleration();
    output_results(out_step);

    // main time loop
    while(time < end_time - dt/2.)
    {
        ++step;
        time += dt;
        update_force();
        assemble_system();
        intermediate_step();
        update_step(); 

        pcout << " Time = " << time << "\n";
        pcout << "   Initial residual at time step " << step << ": "  << constrained_residual.l2_norm() << "\n";

        unsigned int count = 0;

        while(count < 10 && constrained_residual.l2_norm() > rtol)
        {
            solve();
            local_displacement.add(1., newton_update);
            displacement = local_displacement;

            update_force();
            assemble_system();
            update_step();
            ++count;
            pcout << "     Residual after newton step " << count << 
                     ": " << constrained_residual.l2_norm() << "\n";
        }

        pcout << "   Final residual at time step " << step << ": "  << constrained_residual.l2_norm() << "\n\n";
       
        if(step % 20 == 0)
        {
            ++out_step; 
            output_results(out_step);
        }
    }
}

int main(int argc, char** argv)
{
    // set up MPI communicator
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    MPI_Comm mpi_communicator = MPI_COMM_WORLD;    

    // make triangulation
    const int n_global_refinements = 2;
    parallel::shared::Triangulation<3> triangulation(mpi_communicator);
    make_triangulation(triangulation, n_global_refinements);   
 
    // run the model using hexes
    implicit_beam<3> implicit_test(triangulation,
                                   2, 
                                   5.);
    implicit_test.run();


    // run the model with simplices
    parallel::shared::Triangulation<3> simplex_triangulation(mpi_communicator);
    GridGenerator::convert_hypercube_to_simplex_mesh(triangulation, simplex_triangulation);
    implicit_beam<3> simplex_test(simplex_triangulation,
                                  2,
                                  5.);
    simplex_test.run(); 

    return 0;
}
