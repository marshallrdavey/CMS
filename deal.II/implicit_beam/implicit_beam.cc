#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/solver_gmres.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_out.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/constrained_linear_operator.h>

#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/numerics/error_estimator.h>

#include <deal.II/physics/elasticity/kinematics.h>

#include <math.h>

// This is a program to deform a beam with body force in the absence of momentum.
// A 2-dimensional implementation is not supported in this model.

using namespace dealii;

// class declaration
template <int dim>
class implicit_beam
{
public:
    implicit_beam(const unsigned int fe_order, 
                const unsigned int n_global_refinements,
                const double alpha);

    void run();

private:
    void setup_mesh();
    void setup_system();
    void assemble_mass_matrix();
    void update_force();
    void initialize_acceleration();
    void intermediate_step();
    void update_step();
    void assemble_system();
    void solve();
    void output_results(const unsigned int& step) const;

    unsigned int fe_order;
    unsigned int n_global_refinements;
    
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

    // mesh
    Triangulation<dim> triangulation;

    // fe stuff
    FESystem<dim>   fe;
    DoFHandler<dim> dof_handler;

    // constraints
    AffineConstraints<double> constraints;

    // matrix assembly
    SparsityPattern      constrained_sparsity_pattern;
    SparsityPattern      unconstrained_sparsity_pattern;
    SparseMatrix<double> constrained_mass_matrix;
    SparseMatrix<double> unconstrained_mass_matrix;
    SparseMatrix<double> constrained_system_matrix;
    SparseMatrix<double> unconstrained_system_matrix;

    // solution vectors
    Vector<double> displacement;        // total displacement
    Vector<double> displacement_tilde;  // intermediate displacement
    Vector<double> velocity;            // current velocity 
    Vector<double> velocity_tilde;      // intermediate velocity
    Vector<double> acceleration;        // current acceleration
    Vector<double> force;               // current internal and external force
    Vector<double> newton_update;       // displacement update from the newton iteration
    Vector<double> residual;            // residual
    Vector<double> constrained_residual;// constrained residual, system rhs
    const double rtol = 1.e-8;          // residual norm tolerance for Newton's method

    // J storage
    Vector<double> J_vector;
};

// class constructor
template <int dim>
implicit_beam<dim>::implicit_beam(const unsigned int fe_order,
                                  const unsigned int n_global_refinements,
                                  const double alpha)
    :fe_order(fe_order),
     n_global_refinements(n_global_refinements),
     alpha(alpha),
     fe(FE_Q<dim>(fe_order), dim),
     dof_handler(triangulation)
{}

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

// script_A from the deformation dradient, d^2W/dFF^2
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

// Generate beam
template <int dim>
void implicit_beam<dim>::setup_mesh()
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
}

// dof handler stuff
template <int dim>
void implicit_beam<dim>::setup_system()
{
    dof_handler.distribute_dofs(fe);

    // solution vectors
    displacement.reinit(dof_handler.n_dofs());
    displacement_tilde.reinit(dof_handler.n_dofs());
    velocity.reinit(dof_handler.n_dofs());
    velocity_tilde.reinit(dof_handler.n_dofs());
    acceleration.reinit(dof_handler.n_dofs());
    force.reinit(dof_handler.n_dofs());
    newton_update.reinit(dof_handler.n_dofs());
    residual.reinit(dof_handler.n_dofs());
    constrained_residual.reinit(dof_handler.n_dofs());

    // J storage
    J_vector.reinit(triangulation.n_active_cells());

    // constrain lower z boundary to 0 displacement
    constraints.clear();
    VectorTools::interpolate_boundary_values(dof_handler,
                                             4,
                                             Functions::ZeroFunction<dim>(dim),
                                             constraints);
    constraints.close();

    // constrained matrix generation
    DynamicSparsityPattern constrained_dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler,
                                    constrained_dsp,
                                    constraints,
                                    false); //  keep constrained dofs

    constrained_sparsity_pattern.copy_from(constrained_dsp);
    constrained_mass_matrix.reinit(constrained_sparsity_pattern);
    constrained_system_matrix.reinit(constrained_sparsity_pattern);

    // unconstrained mass matrix
    DynamicSparsityPattern unconstrained_dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler,
                                    unconstrained_dsp);
    unconstrained_sparsity_pattern.copy_from(unconstrained_dsp);
    unconstrained_mass_matrix.reinit(unconstrained_sparsity_pattern);
    unconstrained_system_matrix.reinit(unconstrained_sparsity_pattern);
}

// assemble the mass matrix, does not change over time
template<int dim>
void implicit_beam<dim>::assemble_mass_matrix()
{
    constrained_mass_matrix = 0;
    unconstrained_mass_matrix = 0;

    // choose quadrature rule
    const QGauss<dim> quadrature_formula(fe.degree+1);

    // set up FEValues.
    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values |  
                            update_JxW_values);

    // grab the dofs per cell, dim*nodes
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

    // initialize local matrix
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);

    // dof vector for global indices
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // loop over cells
    for (const auto &cell : dof_handler.active_cell_iterators())
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
                const unsigned int i_component = fe.system_to_component_index(i).first; 
                // loop over dof indices
                for(const unsigned int j: fe_values.dof_indices())
                {
                    const unsigned int j_component = fe.system_to_component_index(j).first;
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

// update the force vector
template<int dim>
void implicit_beam<dim>::update_force()
{
    force = 0;
    J_vector = 0;

    // choose quadrature rule
    const QGauss<dim> quadrature_formula(fe.degree+1);
    const QGauss<dim-1> quadrature_formula_face(fe.degree+1);

    // set up FEValues.
    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients | 
                            update_JxW_values);

    // set up FEFaceValues
    FEFaceValues<dim> fe_face_values(fe,
                                     quadrature_formula_face,
                                     update_values | update_normal_vectors | 
                                     update_JxW_values);

    // extractor for the displacement values
    FEValuesExtractors::Vector  u_fe(0);
    std::vector<Tensor<2, dim>> qp_Grad_u;     

    // grab the dofs per cell, dim*nodes
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

    // initialize local rhs
    Vector<double> cell_rhs(dofs_per_cell);

    // dof vector for global indices
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // J value stuff
    double new_volume;

    // loop over cells
    for (const auto &cell : dof_handler.active_cell_iterators())
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
                const unsigned int i_component = fe.system_to_component_index(i).first; 
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
                        const unsigned int i_component = fe.system_to_component_index(i).first;
                        cell_rhs(i) += pressure*
                                       pressure_vector[i_component]*
                                       fe_face_values.shape_value(i, f_q_index)*
                                       fe_face_values.JxW(f_q_index); 
                    }                     
                } 
                
            }
        } 
   
        cell->get_dof_indices(local_dof_indices);
        force.add(local_dof_indices, cell_rhs);
        J_vector[cell->active_cell_index()] = new_volume/cell->measure();
    }
}

// initialize acceleration, M a = f
template<int dim>
void implicit_beam<dim>::initialize_acceleration()
{
    SolverControl            solver_control(1000, 1e-12);
    SolverCG<Vector<double>> solver(solver_control);

    PreconditionJacobi<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(constrained_mass_matrix);
   
    // constrain the force
    auto force_system_operator = linear_operator(unconstrained_mass_matrix);
    auto setup_constrained_force = constrained_right_hand_side(
        constraints, force_system_operator, force);
    Vector<double> force_rhs(dof_handler.n_dofs());
    setup_constrained_force.apply(force_rhs);

    solver.solve(constrained_mass_matrix, acceleration, force_rhs, preconditioner);

    constraints.distribute(acceleration);
}

// compute velocity_tilde and displacement_tilde
template<int dim>
void implicit_beam<dim>::intermediate_step()
{
    // intermediate displacment
    displacement_tilde = 0;
    displacement_tilde.add(1.0, displacement);
    displacement_tilde.add(dt, velocity);
    displacement_tilde.add(dt*dt*(1.0-2.0*beta)/2.0, acceleration);

    // intermediate velocity
    velocity_tilde = 0;
    velocity_tilde.add(1.0, velocity);
    velocity_tilde.add(dt*(1.0-gamma), acceleration);
}

// update velocity, acceleration, and the residual
template<int dim>
void implicit_beam<dim>::update_step()
{
    // update acceleration
    acceleration = 0;
    acceleration.add(1.0/beta/dt/dt, displacement);
    acceleration.add(-1.0/beta/dt/dt, displacement_tilde);
    
    // update velocity
    velocity = 0;
    velocity.add(1.0, velocity_tilde);
    velocity.add(gamma*dt, acceleration);

    // update the residual
    residual = 0;
    residual.add(1.0, force);
    const Vector<double> negative_acceleration = -1.0*acceleration;
    unconstrained_mass_matrix.vmult_add(residual, negative_acceleration);
    residual.add(-0.5*eta_damping, velocity_tilde);
    residual.add(-0.5*eta_damping, velocity);    

    // constrain residual
    constrained_residual = 0;
    auto residual_system_operator = linear_operator(unconstrained_system_matrix);
    auto setup_constrained_residual = constrained_right_hand_side(
      constraints, residual_system_operator, residual);
    setup_constrained_residual.apply(constrained_residual);
}

// matrix assembly
template <int dim>
void implicit_beam<dim>::assemble_system()
{
    constrained_system_matrix = 0;
    unconstrained_system_matrix = 0;

    // choose quadrature rule
    const QGauss<dim> quadrature_formula(fe.degree+1);

    // set up FEValues.
    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients | 
                            update_JxW_values);

    // extractor for the displacement values
    FEValuesExtractors::Vector  u_fe(0);
    std::vector<Tensor<2, dim>> qp_Grad_u;     

    // grab the dofs per cell, dim*nodes
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

    // initialize local matrix and rhs
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);

    // dof vector for global indices
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
 
    // loop over cells
    for (const auto &cell : dof_handler.active_cell_iterators())
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
            // d^2W/dFF      
            const Tensor<4, dim> AA = script_A(FF, I1, J);

            // loop over dof indices
            for(const unsigned int i: fe_values.dof_indices())
            {
                // grad u dimension of dof i
                const unsigned int i_component = fe.system_to_component_index(i).first;
                // loop over dof indices
                for(const unsigned int j: fe_values.dof_indices())
                {
                    // grad u dimension of dof j
                    const unsigned int j_component = fe.system_to_component_index(j).first;

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

// matrix solve
template <int dim>
void implicit_beam<dim>::solve()
{
    SolverControl            solver_control(1000, 1e-12);
    SolverGMRES<Vector<double>> solver(solver_control);

    PreconditionJacobi<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(constrained_system_matrix);
    
    newton_update = 0;
    solver.solve(constrained_system_matrix, newton_update, constrained_residual, preconditioner);

    constraints.distribute(newton_update);
}

// ouptut function
template <int dim>
void implicit_beam<dim>::output_results(const unsigned int& step) const
{
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);

    std::vector<std::string> displacement_names;
    displacement_names.emplace_back("x_displacement");
    displacement_names.emplace_back("y_displacement");
    displacement_names.emplace_back("z_displacement");

    std::vector<std::string> velocity_names;
    velocity_names.emplace_back("x_velocity");
    velocity_names.emplace_back("y_velocity");
    velocity_names.emplace_back("z_velocity");

    std::vector<std::string> acceleration_names;
    acceleration_names.emplace_back("x_acceleration");
    acceleration_names.emplace_back("y_acceleration");
    acceleration_names.emplace_back("z_acceleration");

    data_out.add_data_vector(displacement, displacement_names);
    data_out.add_data_vector(velocity, velocity_names);
    data_out.add_data_vector(acceleration, acceleration_names);
    data_out.add_data_vector(J_vector, "J");

    data_out.set_flags(DataOutBase::VtkFlags(time, step));

    data_out.build_patches();

    std::ofstream output("solution-" + std::to_string(step) + ".vtk");
    data_out.write_vtk(output);
}


// run function
template <int dim>
void implicit_beam<dim>::run()
{
    setup_mesh();
    std::cout << " Number of active cells:       "
              << triangulation.n_active_cells() << std::endl;
    setup_system();
    std::cout << " Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl;

    std::cout << " Maximal cell diameter: " << GridTools::maximal_cell_diameter(triangulation)
              << "\n\n";

    assemble_mass_matrix();

    // initialize problem
    time = 0.;
    unsigned int step = 0;
    unsigned int out_step = 0;
    update_force();
    initialize_acceleration();
    output_results(step);

    while(time < end_time - dt/2.)
    {
        ++step;
        time += dt;
        update_force();
        assemble_system();
        intermediate_step();
        update_step(); 

        std::cout << " Time = " << time << "\n";
        std::cout << "   Initial residual at time step " << step << ": "  << constrained_residual.l2_norm() << "\n";

        unsigned int count = 0;

        while(count < 10 && constrained_residual.l2_norm() > rtol)
        {
            solve();
            displacement.add(1., newton_update);

            update_force();
            assemble_system();
            update_step();
            ++count;
            std::cout << "     Residual after newton step " << count << 
                         ": " << constrained_residual.l2_norm() << "\n";
        }

        std::cout << "   Final residual at time step " << step << ": "  << constrained_residual.l2_norm() << "\n\n";
       
        if(step % 20 == 0)
        {
            ++out_step; 
            output_results(out_step);
        }
    }
}

int main()
{
    implicit_beam<3> implicit_test(2, 2, 5.);
    implicit_test.run();

    return 0;
}
