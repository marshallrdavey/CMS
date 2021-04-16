#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
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
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>

#include <math.h>

using namespace dealii;

// This program solve the convection-diffuction equation
//
//    - \epsilon*Laplacian(u) + \omega\dot\nabla(u) = f


// forcing function 
template <int dim>
class ForcingFunction : public Function<dim>
{
public:
    ForcingFunction()
        : Function<dim>(1)
    {}

    virtual double
    value(const Point<dim> &p, unsigned int component = 0) const override
    {
        (void)component;

        const double alpha = 5.;
        const double epsilon = 1.e-8;
        const double denom = 1. - std::exp(1./alpha);
    
        double x = 0.;

        if(dim == 1)
        {
            x -= epsilon*std::exp(p(0)/alpha)/alpha/alpha/denom;
            x += 1. + std::exp(p(0)/alpha)/alpha/denom;
            return x;
        }
        else if(dim == 2)
        {
            x -= epsilon*((std::exp(p(1)/alpha)*(-((1 - std::exp(p(0)/alpha))/(1 - std::exp(1/alpha))) + p(0)))/(alpha*alpha*(1 - std::exp(1/alpha))) + 
    (std::exp(p(0)/alpha)*(-((1 - std::exp(p(1)/alpha))/(1 - std::exp(1/alpha))) + p(1)))/(alpha*alpha*(1 - std::exp(1/alpha))));
            x += (1 + std::exp(p(1)/alpha)/(alpha*(1 - std::exp(1/alpha))))*p(0)*(-((1 - std::exp(p(0)/alpha))/(1 - std::exp(1/alpha))) + p(0)) - 
    (1 + std::exp(p(0)/alpha)/(alpha*(1 - std::exp(1/alpha))))*p(1)*(-((1 - std::exp(p(1)/alpha))/(1 - std::exp(1/alpha))) + p(1));
            return x;
        } 
        else 
        {
            return x;
        }
    
    }
};


// exact function
template <int dim>
class ExactSolution : public Function<dim>
{
public:
    ExactSolution()
        : Function<dim>(1)
    {}

    virtual double
    value(const Point<dim> &p, unsigned int component = 0) const override
    {
        (void)component;

        const double alpha = 5.;
        double x;
        if(dim == 1)
        {       
            x = (p(0) - (1. - std::exp(p(0)/alpha))/(1. - std::exp(1./alpha)));
            return x;
        }
        else if(dim == 2)
        {
            x = (p(0) - (1. - std::exp(p(0)/alpha))/(1. - std::exp(1./alpha)))*
                (p(1) - (1. - std::exp(p(1)/alpha))/(1. - std::exp(1./alpha)));
            return x;
        }
        else
        {
            return x;
        }
    } 
};


template <int dim>
class SUPG
{
public:
    SUPG(const unsigned int n_global_refinements,
         const bool do_supg);

    void run();

private:
    void setup_grid();
    void setup_system();
    void assemble_system_matrix();
    void assemble_rhs();
    void solve();
    void output_results() const;

    unsigned int n_global_refinements;
    bool do_supg;

    Triangulation<dim> triangulation;

    FE_Q<dim>       finite_element;
    DoFHandler<dim> dof_handler;

    // parameters
    const double alpha = 5.;
    const double epsilon = 1.e-8;
    double delta;  
    Tensor<1, dim> omega(const Point<dim> &p);

    AffineConstraints<double> homogeneous_constraints;

    SparsityPattern      constrained_sparsity_pattern;
    SparseMatrix<double> constrained_system_matrix;

    Vector<double> solution;
    Vector<double> system_rhs;
};

// class constructor
template <int dim>
SUPG<dim>::SUPG(const unsigned int n_global_refinements,
                const bool do_supg)
  : n_global_refinements(n_global_refinements),
    do_supg(do_supg),  
    finite_element(1),
    dof_handler(triangulation)
{
    static_assert(dim < 3, "The dimension must be less than 3 because David doesn't trust 3D space.\n"
                           "I must assume this is because he is a flat Earth truther.\n");
}

template<int dim>
Tensor<1, dim> SUPG<dim>::omega(const Point<dim>& p)
{
    Tensor<1, dim> tensor;
    if(dim == 1)
    {
        tensor[0] = 1.0;
        return tensor;
    }
    else if(dim == 2)
    {
        tensor[0] = -p(1);
        tensor[1] = p(0);
        return tensor;
    }  
    else 
    {
        return tensor;
    }
}

// construct triangulation 
template <int dim>
void SUPG<dim>::setup_grid()
{
    // hyper cube generation
    GridGenerator::hyper_cube(triangulation,
                              0.0, // left bound
                              1.0, // right bound
                              true); // colorize

    triangulation.refine_global(n_global_refinements);

    if(do_supg) 
    {
        delta = GridTools::minimal_cell_diameter(triangulation);
        std::cout << " This run is using SUPG.\n";
    }
    else
    {
        delta = 0.0;
        std::cout << " This run is not using SUPG.\n";
    }
}

// setup system and dofs 
template <int dim>
void SUPG<dim>::setup_system()
{
    dof_handler.distribute_dofs(finite_element);

    // construct solution vector
    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());

    // boundary constraints
    homogeneous_constraints.clear();
    std::vector<types::boundary_id> bdry_ids = triangulation.get_boundary_ids();
    for(unsigned int i = 0; i < bdry_ids.size(); ++i)
    {
        VectorTools::interpolate_boundary_values(dof_handler,
                                                 bdry_ids[i],
                                                 ExactSolution<dim>(),
                                                 homogeneous_constraints);
    }  
    homogeneous_constraints.close();

    // matrix construction
    DynamicSparsityPattern constrained_dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler,
                                    constrained_dsp,
                                    homogeneous_constraints,
                                    false);
    constrained_sparsity_pattern.copy_from(constrained_dsp);
    constrained_system_matrix.reinit(constrained_sparsity_pattern);
}


// assemble system matrix
template <int dim>
void SUPG<dim>::assemble_system_matrix()
{
    constrained_system_matrix = 0;
    QGauss<dim> quadrature(finite_element.degree + 1);

    // Set up FEValues.
    FEValues<dim> fe_values(finite_element,
                            quadrature,
                            update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);
    // grab dofs per cell
    const unsigned int dofs_per_cell = finite_element.n_dofs_per_cell();

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
            const Tensor<1, dim> omega_q = omega(fe_values.quadrature_point(q_index));
            // loop over test functions
            for (const unsigned int i : fe_values.dof_indices())
            {
                // loop over trial functions
                for (const unsigned int j : fe_values.dof_indices())
                {
                    cell_matrix(i, j) += (epsilon*(fe_values.shape_grad(i, q_index)*
                                                   fe_values.shape_grad(j, q_index)) + 
                                          fe_values.shape_value(i, q_index)*
                                          (omega_q*fe_values.shape_grad(j, q_index)) +
                                          delta*(omega_q*fe_values.shape_grad(i, q_index))*
                                                (omega_q*fe_values.shape_grad(j, q_index)))*
                                         fe_values.JxW(q_index);  
                }
            }
        }
        // distribute local matrix to global matrix
        cell->get_dof_indices(local_dof_indices);
        homogeneous_constraints.distribute_local_to_global(
            cell_matrix, local_dof_indices, constrained_system_matrix);
    }
}

// assemble system rhs
template <int dim>
void SUPG<dim>::assemble_rhs()
{
    system_rhs = 0;
    QGauss<dim> quadrature(finite_element.degree + 1);

    // Set up FEValues.
    FEValues<dim> fe_values(finite_element,
                            quadrature,
                            update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);
    // grab dofs per cell
    const unsigned int dofs_per_cell = finite_element.n_dofs_per_cell();

    // initialize local matrix 
    Vector<double> cell_rhs(dofs_per_cell);

    // dof vector for global indices
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // 
    const ForcingFunction<dim> forcing_function;

    // loop over cells
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        cell_rhs = 0;
        fe_values.reinit(cell);
      
        // loop over quadrature points
        for (const unsigned int q_index: fe_values.quadrature_point_indices())
        {
            const Tensor<1, dim> omega_q = omega(fe_values.quadrature_point(q_index));
            const double force_q = forcing_function.value(fe_values.quadrature_point(q_index));
            // loop over test functions
            for (const unsigned int i : fe_values.dof_indices())
            {
                cell_rhs(i) += force_q*(fe_values.shape_value(i, q_index) + 
                                        delta*(omega_q*fe_values.shape_grad(i, q_index)))*                                    
                               fe_values.JxW(q_index);  
            }
        }
        // distribute local matrix to global matrix
        cell->get_dof_indices(local_dof_indices);
        homogeneous_constraints.distribute_local_to_global(
            cell_rhs, local_dof_indices, system_rhs);
    }
}


// intialize a time step
template <int dim>
void SUPG<dim>::solve()
{
    SolverControl            solver_control(10000, 1e-12);
    SolverGMRES<Vector<double>> solver(solver_control);

    PreconditionJacobi<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(constrained_system_matrix);

    solver.solve(constrained_system_matrix, solution, system_rhs, preconditioner);
    homogeneous_constraints.distribute(solution);
}

// output solution files
template <int dim>
void SUPG<dim>::output_results() const
{
    // Compute the pointwise maximum error:
    Vector<double> max_error_per_cell(triangulation.n_active_cells());
    ExactSolution<dim> exact_solution;

    MappingQGeneric<dim> mapping(1);
    VectorTools::integrate_difference(mapping,
                                      dof_handler,
                                      solution,
                                      exact_solution,
                                      max_error_per_cell,
                                      QIterated<dim>(QGauss<1>(2), 2),
                                      VectorTools::NormType::Linfty_norm);
    std::cout << "  maximum error = "
              << *std::max_element(max_error_per_cell.begin(),
                                   max_error_per_cell.end())
              << std::endl;

    // Save the output:
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "solution");
    data_out.add_data_vector(max_error_per_cell, "max_error_per_cell");
    data_out.build_patches();
    if(do_supg)
    {
        std::ofstream output("output-" + std::to_string(dim) + "d-" +
                             std::to_string(n_global_refinements) + "r-" +
                             "supg.vtu");
        data_out.write_vtu(output);
    }
    else
    {

        std::ofstream output("output-" + std::to_string(dim) + "d-" +
                             std::to_string(n_global_refinements) + "r-" +
                             "no_supg.vtu");
        data_out.write_vtu(output);
    }
}

// run everything
template <int dim>
void SUPG<dim>::run()
{
    setup_grid();
    std::cout << "   Number of active cells:       "
              << triangulation.n_active_cells() << std::endl;

    setup_system();
    std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
              << "\n";

    std::cout << "   Maximal cell diameter: " << GridTools::maximal_cell_diameter(triangulation)
              << std::endl;

    assemble_system_matrix();
    assemble_rhs();

    solve();
    output_results();
}


int main()
{
    // 1d loop
    for (unsigned int i = 1; i < 8; ++i)
    {
        SUPG<1> supg(i, true);
        supg.run();
    }
    
    // 2d loop
    for (unsigned int i = 1; i < 8; ++i)
    {
        SUPG<2> supg(i, true);
        supg.run();
    }

  return 0;
}
