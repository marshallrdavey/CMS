#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/quadrature_lib.h>
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
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/physics/elasticity/kinematics.h>

#include <fstream>
#include <math.h>

// This is a program to deform a beam with body force and/or a traction force
// with an explicit time stepping scheme.
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
class explicit_beam
{
public:
    explicit_beam(const parallel::shared::Triangulation<dim>& triangulation,
                  const unsigned int fe_order, 
                  const double alpha);

    void run();

private:
    // class functions
    void setup_system();
    void assemble_system_matrix();
    void assemble_rhs();
    void update_residual();
    void solve();
    void output_results(const unsigned int& step);

    // MPI communicator
    MPI_Comm mpi_comm;
    ConditionalOStream pcout;

    // mesh and finite element parameters
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
    const double rho = 1.; 
    const double beta = 1.e3;
    const double eta_damping = 0.0003; 
    double I_1(const Tensor<2, dim>& FF);
    Tensor<2, dim> PK1_stress(const Tensor<2, dim>& FF,
                              const double& I1,
                              const double& J);
    
    // body force and pressure
    const Vector<double> body_force{0., 0.001, 0.};
    const double pressure = 0.;
    const Vector<double> pressure_vector{0., 0., 0.};

    // time
    const double dt = 0.002;
    const double end_time = 80.;
    double time;

    // constraint
    AffineConstraints<double> constraints;

    // matrix things
    la::MPI::SparseMatrix constrained_system_matrix;
    la::MPI::SparseMatrix unconstrained_system_matrix;

    // solution vectors
    la::MPI::Vector displacement;    
    la::MPI::Vector local_displacement;    
    la::MPI::Vector velocity;           
    la::MPI::Vector local_velocity;           
    la::MPI::Vector acceleration;    
    la::MPI::Vector local_acceleration;    
    la::MPI::Vector residual;
    la::MPI::Vector local_residual;
    la::MPI::Vector constrained_residual;

    // J storage vector
    la::MPI::Vector J_vector;       
};

// class constructor
template <int dim>
explicit_beam<dim>::explicit_beam(const parallel::shared::Triangulation<dim>& triangulation,
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

// I_1, tr(CC)
template <int dim>
double explicit_beam<dim>::I_1(const Tensor<2, dim>& FF)
{
    return trace(Physics::Elasticity::Kinematics::C(FF));
}

// PK1_stress, PP, dW/dFF
template <int dim>
Tensor<2, dim> explicit_beam<dim>::PK1_stress(const Tensor<2, dim>& FF,
                                              const double& I1,
                                              const double& J)
{
    return alpha/cbrt(J*J)*(FF - I1*transpose(invert(FF))/3.);
}

// dof handler stuff
template <int dim>
void explicit_beam<dim>::setup_system()
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

    velocity.reinit(locally_owned_dofs,
                    locally_relevant_dofs,
                    mpi_comm);
    local_velocity.reinit(locally_owned_dofs, mpi_comm);    

    acceleration.reinit(locally_owned_dofs,
                        locally_relevant_dofs,
                        mpi_comm);
    local_acceleration.reinit(locally_owned_dofs, mpi_comm);    

    residual.reinit(locally_owned_dofs,
                    locally_relevant_dofs,
                    mpi_comm);
    local_residual.reinit(locally_owned_dofs, mpi_comm);    
    constrained_residual.reinit(locally_owned_dofs, mpi_comm);    

    // J vector
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
    constrained_system_matrix.reinit(locally_owned_dofs,
                                     locally_owned_dofs,
                                     constrained_dsp,
                                     mpi_comm);

    // unconstrained matrix generation
    DynamicSparsityPattern unconstrained_dsp(locally_relevant_dofs);
    DoFTools::make_sparsity_pattern(dof_handler,
                                    unconstrained_dsp);
    SparsityTools::distribute_sparsity_pattern(unconstrained_dsp,
                                               locally_owned_dofs,
                                               mpi_comm,
                                               locally_relevant_dofs);
    unconstrained_system_matrix.reinit(locally_owned_dofs,
                                       locally_owned_dofs,
                                       unconstrained_dsp,
                                       mpi_comm);
}

// matrix assembly
// this is only called once as the matrix is only assembled once
template <int dim>
void explicit_beam<dim>::assemble_system_matrix()
{
    constrained_system_matrix = 0;
    unconstrained_system_matrix = 0;

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
                                              0.);
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

// rhs assembly
template <int dim>
void explicit_beam<dim>::assemble_rhs()
{
    local_residual = 0;
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
            local_residual.add(local_dof_indices, cell_rhs);
            J_vector[cell->active_cell_index()] = new_volume/cell->measure();
        }
    }
    local_residual.compress(VectorOperation::add);
    J_vector.compress(VectorOperation::insert);
}

// update residual
template <int dim>
void explicit_beam<dim>::update_residual()
{
    // add force from velocity damping
    local_residual.add(-eta_damping, local_velocity);
    residual = local_residual;
    
    // constrain residual 
    constrained_residual = 0;
    auto system_operator = 
        linear_operator<la::MPI::Vector>(unconstrained_system_matrix);
    auto setup_constrained_residual = constrained_right_hand_side<la::MPI::Vector>(
        constraints, system_operator, residual);
    setup_constrained_residual.apply(constrained_residual);
}

// matrix solve
template <int dim>
void explicit_beam<dim>::solve()
{
    // solver settings
    SolverControl solver_control(1000, 1e-12);
    la::SolverCG  solver(solver_control);

    // preconditioner settings
    la::MPI::PreconditionAMG preconditioner;
    la::MPI::PreconditionAMG::AdditionalData data;
    data.symmetric_operator = true;
    preconditioner.initialize(constrained_system_matrix, data);
  
    // solve for updated acceleration 
    local_acceleration = 0;
    solver.solve(constrained_system_matrix, local_acceleration, constrained_residual, preconditioner);

    // constrain accleration
    constraints.distribute(local_acceleration);
}

// ouptut function
template <int dim>
void explicit_beam<dim>::output_results(const unsigned int& step)
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

    // update output vectors to ghosted vectors
    displacement = local_displacement;
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
void explicit_beam<dim>::run()
{
    pcout << "   Number of active cells:       "
              << tria->n_active_cells() << std::endl;
    setup_system();
    pcout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl;

    pcout << "   Maximal cell diameter: " << GridTools::maximal_cell_diameter(*tria)
              << std::endl;

    assemble_system_matrix();
    unsigned int step = 0;
    unsigned int out_step = 0;
    time = 0.;

    // initial step
    pcout << "   Time: " << time << "\n";
    assemble_rhs();
    update_residual();


    pcout << "   Residual at step " << step << ": "  << constrained_residual.l2_norm() << "\n";
    solve();
    output_results(out_step);

    // main time loop 
    while(time < end_time - 0.5*dt)
    {
        ++step;
        time += dt;        
        pcout << "   Time: " << time << "\n";

        // v_n+1/2 = v_n + a_n*dt/2
        local_velocity.add(dt/2.0, local_acceleration);
        // constrain v
        constraints.distribute(local_velocity);

        // d_n+1 = d_n + dt*v_n+1/2
        local_displacement.add(dt, local_velocity);
        displacement = local_displacement;

        // compute force and solve for acceleration 
        assemble_rhs();
        update_residual();
        pcout << "   Residual at step " << step << ": "  << constrained_residual.l2_norm() << "\n";
        solve();
        
        // v_n+1 = v_n+1/2 + a_n+1*dt/2
        local_velocity.add(dt/2.0, local_acceleration);
        if(step % 1000 == 0)
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
    explicit_beam<3> explicit_test(triangulation,
                                   2, 
                                   5.);
    explicit_test.run();


    // run the model with simplices
    parallel::shared::Triangulation<3> simplex_triangulation(mpi_communicator);
    GridGenerator::convert_hypercube_to_simplex_mesh(triangulation, simplex_triangulation);
    explicit_beam<3> simplex_test(simplex_triangulation,
                                  2,
                                  5.);
    simplex_test.run(); 
}
