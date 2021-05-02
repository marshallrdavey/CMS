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

#include <deal.II/grid/grid_tools.h>

#include <deal.II/numerics/error_estimator.h>

#include <deal.II/physics/elasticity/kinematics.h>

#include <math.h>

// This is a program to deform a beam with body force and/or a traction force
// with an explicit time stepping scheme.
// A 2-dimensional implementation is not supported in this model.

using namespace dealii;

// class declaration
template <int dim>
class explicit_beam
{
public:
    explicit_beam(const unsigned int fe_order, 
                const unsigned int n_global_refinements,
                const double alpha);

    void run();

private:
    void setup_mesh();
    void setup_system();
    void assemble_system_matrix();
    void assemble_rhs();
    void solve();
    void output_results(const unsigned int& step) const;

    unsigned int fe_order;
    unsigned int n_global_refinements;
    
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

    // mesh
    Triangulation<dim> triangulation;

    // finite element handler
    FESystem<dim>   fe;
    DoFHandler<dim> dof_handler;

    // constraint
    AffineConstraints<double> constraints;

    // matrix things
    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;

    // solution vectors
    Vector<double> displacement;    
    Vector<double> velocity;           
    Vector<double> acceleration;    
    Vector<double> residual;

    // J vector
    Vector<double> J_vector;       
};

// class constructor
template <int dim>
explicit_beam<dim>::explicit_beam(const unsigned int fe_order,
                              const unsigned int n_global_refinements,
                              const double alpha)
    :fe_order(fe_order),
     n_global_refinements(n_global_refinements),
     alpha(alpha),
     fe(FE_Q<dim>(fe_order), dim),
     dof_handler(triangulation)
{}

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

// Generate beam
template <int dim>
void explicit_beam<dim>::setup_mesh()
{
    const Point<3> c0(0, 0, 0);
    const Point<3> c1(1, 1, 2);
    const std::vector<unsigned int> rep{1, 1, 2};  
    GridGenerator::subdivided_hyper_rectangle(triangulation,
                                              rep,          // inumber of cells in each direction
                                              c0,           // origin corner
                                              c1,           // opposite corner
                                              true);        // colorize   

    triangulation.refine_global(n_global_refinements);
}

// dof handler stuff
template <int dim>
void explicit_beam<dim>::setup_system()
{
    dof_handler.distribute_dofs(fe);

    // solution vectors
    displacement.reinit(dof_handler.n_dofs());    
    velocity.reinit(dof_handler.n_dofs());           
    acceleration.reinit(dof_handler.n_dofs());    
    residual.reinit(dof_handler.n_dofs());

    // J vector
    J_vector.reinit(triangulation.n_active_cells());       

    // constrain lower z boundary to 0 displacement
    constraints.clear();
    VectorTools::interpolate_boundary_values(dof_handler,
                                             4,
                                             Functions::ZeroFunction<dim>(dim),
                                             constraints);
    constraints.close();

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler,
                                    dsp,
                                    constraints,
                                    false); //  keep constrained dofs

    sparsity_pattern.copy_from(dsp);
    system_matrix.reinit(sparsity_pattern);
}

// matrix assembly
// this is only called once as the matrix is only assembled once
template <int dim>
void explicit_beam<dim>::assemble_system_matrix()
{
    system_matrix = 0;

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
                                              0.);
                }
            }
        }
        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(
            cell_matrix, local_dof_indices, system_matrix);
    }
}

// rhs assembly
template <int dim>
void explicit_beam<dim>::assemble_rhs()
{
    residual = 0;
    J_vector = 0;

    // choose quadrature rule
    const QGauss<dim> quadrature_formula(fe.degree+1);
    const QGauss<dim - 1> quadrature_formula_face(fe.degree+1);

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
    Vector<double>     cell_rhs(dofs_per_cell);

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
        constraints.distribute_local_to_global(
            cell_rhs, local_dof_indices, residual);
        J_vector[cell->active_cell_index()] = new_volume/cell->measure();
    }
}

// matrix solve
template <int dim>
void explicit_beam<dim>::solve()
{
    SolverControl            solver_control(1000, 1e-12);
    SolverCG<Vector<double>> solver(solver_control);

    PreconditionSSOR<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(system_matrix, 1.2);
   
    residual.add(-eta_damping, velocity); 
    solver.solve(system_matrix, acceleration, residual, preconditioner);

    constraints.distribute(acceleration);
}

// ouptut function
template <int dim>
void explicit_beam<dim>::output_results(const unsigned int& step) const
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

    data_out.build_patches();

    data_out.set_flags(DataOutBase::VtkFlags(time, step));

    std::ofstream output("solution-" + std::to_string(step) + ".vtk");
    data_out.write_vtk(output);
}


// run function
template <int dim>
void explicit_beam<dim>::run()
{
    setup_mesh();
    std::cout << "   Number of active cells:       "
              << triangulation.n_active_cells() << std::endl;
    setup_system();
    std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl;

    std::cout << "   Maximal cell diameter: " << GridTools::maximal_cell_diameter(triangulation)
              << std::endl;

    assemble_system_matrix();
    unsigned int step = 0;
    unsigned int out_step = 0;
    time = 0.;

    // initial step
    std::cout << "   Time: " << time << "\n";
    assemble_rhs();
    std::cout << "   Residual at step " << step << ": "  << residual.l2_norm() << "\n";
    solve();
    output_results(step);
    
    while(time < end_time - 0.5*dt)
    {
        ++step;
        time += dt;        
        std::cout << "   Time: " << time << "\n";

        // v_n+1/2 = v_n + a_n*dt/2
        velocity.add(dt/2.0, acceleration);
        // constrain v
        constraints.distribute(velocity);

        // d_n+1 = d_n + dt*v_n+1/2
        displacement.add(dt, velocity);

        assemble_rhs();
        std::cout << "   Residual at step " << step << ": "  << residual.l2_norm() << "\n";
        solve();
        
        // v_n+1 = v_n+1/2 + a_n+1*dt/2
        velocity.add(dt/2.0, acceleration);
        if(step % 1000 == 0)
        { 
            ++out_step;
            output_results(out_step);
        }
    }
}

int main()
{
    explicit_beam<3> explicit_test(2, 2, 5.);
    explicit_test.run();
    return 0;
}
