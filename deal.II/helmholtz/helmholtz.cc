#include <deal.II/base/quadrature_lib.h>

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

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_out.h>

#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/numerics/error_estimator.h>

#include <math.h>

using namespace dealii;

// -\nabla^2 f + f
template <int dim>
class ForcingFunction : public Function<dim>
{
  public:
        ForcingFunction() : Function<dim>(1) {}

        virtual
        double value(const Point<dim> &p, unsigned int component = 0) const override
        {
            (void)component; // remove compiler warning for unused variable

            // 762: Implement the forcing function here.
            double f = 0.0;
            for(unsigned int i = 0; i < dim; ++i)
            {
                f += (M_PI*M_PI + 1)*sin(M_PI*p(i)); 
            }
            return f;
        }
};

// f = sin(pi*p(i))
template <int dim>
class ExactSolution : public Function<dim>
{
  public:
        ExactSolution() : Function<dim>(1) {}

        virtual
        double value(const Point<dim> &p, unsigned int component = 0) const override
        {
            (void)component;

            // 762: Implement the exact solution here.
            double f = 0.0;
            for(unsigned int i = 0; i < dim; ++i)
            {
                f += sin(M_PI*p(i));
            }
            return f;
        }
};

// class declaration
template <int dim>
class Helmholtz
{
public:
  Helmholtz(const unsigned int n_global_refinements);

  void run();

private:
  void setup_grid();
  void setup_system();
  void assemble_system();
  void solve();
  void output_results() const;

  unsigned int n_global_refinements;

  Triangulation<dim> triangulation;

  FE_Q<dim>       fe;
  DoFHandler<dim> dof_handler;

  AffineConstraints<double> constraints;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;

  Vector<double> solution;
  Vector<double> system_rhs;
};

// class constructor
template <int dim>
Helmholtz<dim>::Helmholtz(const unsigned int n_global_refinements)
  : n_global_refinements(n_global_refinements)
  , fe(2)
  , dof_handler(triangulation)
{}



template <int dim>
void Helmholtz<dim>::setup_grid()
{
    // 762: call a GridGenerator function here. Pick your favorite!
    GridGenerator::plate_with_a_hole(triangulation,
                                     0.5, // inner radius
                                     1.0, // outer radius
                                     1.0, // pad bottom
                                     1.0, // pad top
                                     1.0, // pad left
                                     1.0, // pad right
                                     Point<dim>(), // center
                                     0, // polar manifold id
                                     1, // tfi manifold id
                                     4.0, // L
                                     5, // n slices
                                     false); //colorize   
    triangulation.refine_global(n_global_refinements);
}

// dof handler stuff
template <int dim>
void Helmholtz<dim>::setup_system()
{
  dof_handler.distribute_dofs(fe);

  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());

  // 762: If you want to use periodic or Dirichlet BCs then you should add them
  // here. Take a look at how this is done in step-6 for Dirichlet BCs.
  
  // use Dirichlet BCs with forcing function input on all boundaries
  constraints.clear();
  std::vector<types::boundary_id> bdry_ids = triangulation.get_boundary_ids();
  for(unsigned int i = 0; i < bdry_ids.size(); ++i)
  {
    VectorTools::interpolate_boundary_values(dof_handler,
                                             bdry_ids[i],
                                             ExactSolution<dim>(),
                                             constraints);
  }
  constraints.close();

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler,
                                  dsp,
                                  constraints,
                                  false); //  keep constrained dofs

  sparsity_pattern.copy_from(dsp);
  system_matrix.reinit(sparsity_pattern);
}


// matrix and rhs assembly
template <int dim>
void Helmholtz<dim>::assemble_system()
{
    // 762: Pick a good quadrature formula here.
    const QGauss<dim> quadrature_formula(fe.degree+1);

    // 762: Set up FEValues.
    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients | 
                            update_quadrature_points | update_JxW_values);     

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // initialize forcing function
    const ForcingFunction<dim> forcing_function;

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        cell_matrix = 0;
        cell_rhs    = 0;

        fe_values.reinit(cell);
        // loop over quadrature points
        for (const unsigned int q_index: fe_values.quadrature_point_indices())
        {
            // loop over test functions
            for (const unsigned int i : fe_values.dof_indices())
            {
                // loop over trial functions
                for (const unsigned int j : fe_values.dof_indices())
                {
                    // 762: compute the values of the cell matrix.
                    // (grad phi_i grad phi_j + phi_i phi_j)dx 
                    cell_matrix(i, j) += (fe_values.shape_grad(i, q_index)*
                                          fe_values.shape_grad(j, q_index) + 
                                          fe_values.shape_value(i, q_index)*
                                          fe_values.shape_value(j, q_index))*
                                          fe_values.JxW(q_index); 
                }
                // 762: compute the values of the cell RHSi.
                // phi_i f dx
                cell_rhs(i) += fe_values.shape_value(i, q_index)*
                               forcing_function.value(fe_values.quadrature_point(q_index))*
                               fe_values.JxW(q_index);
            }
        }

        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(
        cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
    }
}

// matrix solve
template <int dim>
void Helmholtz<dim>::solve()
{
  SolverControl            solver_control(1000, 1e-12);
  SolverCG<Vector<double>> solver(solver_control);

  PreconditionSSOR<SparseMatrix<double>> preconditioner;
  preconditioner.initialize(system_matrix, 1.2);

  solver.solve(system_matrix, solution, system_rhs, preconditioner);

  constraints.distribute(solution);
}


// ouptut function
template <int dim>
void Helmholtz<dim>::output_results() const
{
  // Compute the pointwise maximum error:
  Vector<double> max_error_per_cell(triangulation.n_active_cells());
    {
      MappingQGeneric<dim> mapping(1);
      VectorTools::integrate_difference(mapping,
                                        dof_handler,
                                        solution,
                                        ExactSolution<dim>(),
                                        max_error_per_cell,
                                        QIterated<dim>(QGauss<1>(2), 2),
                                        VectorTools::NormType::Linfty_norm);
      std::cout << "maximum error = " << *std::max_element(max_error_per_cell.begin(),
                                                           max_error_per_cell.end())
                << std::endl;
  }

  // Save the output:
  {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "solution");
    data_out.add_data_vector(max_error_per_cell, "max_error_per_cell");
    data_out.build_patches();

    std::ofstream output("output-" + std::to_string(dim) + "d-"
                         + std::to_string(n_global_refinements)
                         + ".vtu");
    data_out.write_vtu(output);
  }
}


// run function
template <int dim>
void Helmholtz<dim>::run()
{
    setup_grid();
    std::cout << "   Number of active cells:       "
              << triangulation.n_active_cells() << std::endl;

    setup_system();
    std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl;

    std::cout << "   Maximal cell diameter: " << GridTools::maximal_cell_diameter(triangulation)
              << std::endl;

    assemble_system();
    solve();
    output_results();
}


int main()
{
  for (unsigned int i = 0; i < 6; ++i)
    {
      Helmholtz<2> helmholtz(i);
      helmholtz.run();
    }

  for (unsigned int i = 0; i < 4; ++i)
    {
      Helmholtz<3> helmholtz(i);
      helmholtz.run();
    }

  return 0;
}
