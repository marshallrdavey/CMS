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
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>

#include <math.h>

using namespace dealii;

// This program does one step of the heat equation
//
//     u_t = Laplacian(u) + f
//
// with the backward Euler method, discretized with the method of lines:
//
//     M u^{k + 1} - M u^{k} = -dt S u^{k + 1} + dt f^{k + 1}
//
// solving for u^{k + 1} yields
//
//     (M + dt S) u^{k + 1} = M u^{k} + dt f^{k + 1}
//
// here M is the mass matrix and -S is the discretization of the Laplacian
// (i.e., -S is ngative semidefinite, so S is positive semidefinite).

// forcing function: dx/dt - \nabla^2 x 
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
    // 762: Implement the forcing function here. This should be time dependent:
    // see the documentation of FunctionTime for more information on how to do
    // this.
    double x = 0.0;
    for(unsigned int i = 0; i < dim; ++i)
    {
        x += sin(M_PI*p(i));
    }
    return x*(M_PI*M_PI*cos(this->get_time()) - sin(this->get_time()));
  }
};

// exact function: x = cos(t)\sum_0^{dim-1}sin(p(i))
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
    // 762: Implement the exact solution here. The same note on FunctionTime
    // applies here too.
    double x = 0.0;
    for(unsigned int i = 0; i < dim; ++i)
    {
        x += sin(M_PI*p(i));
    }
    return x*cos(this->get_time());
  }
};


template <int dim>
class Heat
{
public:
  Heat(const unsigned int n_global_refinements);

  void
  run();

private:
  void
  setup_grid();
  void
  setup_system();
  void
  assemble_system();
  void
  do_one_time_step();
  void
  output_results() const;

  unsigned int n_global_refinements;
  double       time_step;

  Triangulation<dim> triangulation;
  double             current_time;

  FE_Q<dim>       finite_element;
  DoFHandler<dim> dof_handler;

  AffineConstraints<double> homogeneous_constraints;

  SparsityPattern      constrained_sparsity_pattern;
  SparsityPattern      unconstrained_sparsity_pattern;
  SparseMatrix<double> constrained_system_matrix;
  SparseMatrix<double> unconstrained_mass_matrix;
  SparseMatrix<double> unconstrained_system_matrix;

  Vector<double> previous_solution;
  Vector<double> current_solution;
};

// class constructor
template <int dim>
Heat<dim>::Heat(const unsigned int n_global_refinements)
  : n_global_refinements(n_global_refinements)
  , time_step(std::pow(0.5, n_global_refinements))
  , current_time(0.0)
  , finite_element(1)
  , dof_handler(triangulation)
{}

// construct triangulation 
template <int dim>
void
Heat<dim>::setup_grid()
{
  // 762: call a GridGenerator function here. Pick any grid this doesn't have
  // 270 degree corners - for example, with 'cheese' the solution is
  // insufficiently smooth at internal corners which messes up the overall
  // convergence rate.
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

// 
template <int dim>
void
Heat<dim>::setup_system()
{
  dof_handler.distribute_dofs(finite_element);

  // For this problem we have to solve a more complicated situation with
  // constraints - in general, with the method of manufactured solutions, our
  // boundary conditions may be time dependent. However, we still want to know
  // what our constraints are when we set up the linear system. Hence we set up
  // the linear system with homogeneous constraints and then apply the correct
  // inhomogeneities when we solve the system.
  // Note: I moved this in order to declare the correct solution in 
  // current_solution
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

  previous_solution.reinit(dof_handler.n_dofs());
  current_solution.reinit(dof_handler.n_dofs());

  // 762: you will also have to use the exact solution to set up the initial
  // condition. current_solution is used because this is immediately swapped
  // with previous_solution in do_one_time_step 
  VectorTools::project(dof_handler,
                       homogeneous_constraints,
                       QGauss<dim>(finite_element.degree + 1),
                       ExactSolution<dim>(),
                       current_solution);

  // 762: set up the sparsity pattern, mass, and system matrices here.
  // constrained matrices
  DynamicSparsityPattern constrained_dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler,
                                  constrained_dsp,
                                  homogeneous_constraints,
                                  false);
  constrained_sparsity_pattern.copy_from(constrained_dsp);
  constrained_system_matrix.reinit(constrained_sparsity_pattern);
 
  // unconstrained matrices
  DynamicSparsityPattern unconstrained_dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler,
                                  unconstrained_dsp);
  unconstrained_sparsity_pattern.copy_from(unconstrained_dsp);
  unconstrained_system_matrix.reinit(unconstrained_sparsity_pattern);
  unconstrained_mass_matrix.reinit(unconstrained_sparsity_pattern);
}


template <int dim>
void
Heat<dim>::assemble_system()
{
  QGauss<dim> quadrature(finite_element.degree + 1);

  // unnecessary ?
  // MappingQGeneric<dim> mapping(1);

  // 762: Set up FEValues.
  FEValues<dim> fe_values(finite_element,
                          quadrature,
                          update_values | update_gradients |
                          update_JxW_values);

  const unsigned int dofs_per_cell = finite_element.n_dofs_per_cell();

  FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_system_matrix(dofs_per_cell, dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      cell_mass_matrix = 0;
      cell_system_matrix = 0;
      fe_values.reinit(cell);
      
      // 762: write the loops to fill in the matrices here.
      // loop over quadrature points
      for (const unsigned int q_index: fe_values.quadrature_point_indices())
      {
          // loop over test functions
          for (const unsigned int i : fe_values.dof_indices())
          {
              // loop over trial functions
              for (const unsigned int j : fe_values.dof_indices())
              {
                  cell_system_matrix(i, j) += (fe_values.shape_value(i, q_index)*
                                               fe_values.shape_value(j, q_index) +
                                               fe_values.shape_grad(i, q_index)*
                                               fe_values.shape_grad(j, q_index)*
                                               time_step)*
                                               fe_values.JxW(q_index);  
                  cell_mass_matrix(i, j) += fe_values.shape_value(i, q_index)*
                                            fe_values.shape_value(j, q_index)*
                                            fe_values.JxW(q_index); 
              }
          }
      }
      // set up the three matrices we care about.
      cell->get_dof_indices(local_dof_indices);
      homogeneous_constraints.distribute_local_to_global(
        cell_system_matrix, local_dof_indices, constrained_system_matrix);
      // To set up the RHS we need an unconstrained copy of the system matrix
      // and mass matrix too:
      unconstrained_system_matrix.add(local_dof_indices, cell_system_matrix);
      unconstrained_mass_matrix.add(local_dof_indices, cell_mass_matrix);
    }
}

// intialize a time step
template <int dim>
void
Heat<dim>::do_one_time_step()
{
  // The current solution should be swapped with the previous solution:
  std::swap(previous_solution, current_solution);

  // Set up M u^k + dt f^{k + 1}
  Vector<double> load_vector(dof_handler.n_dofs());
  {
    MappingQGeneric<dim> mapping(1);
    ForcingFunction<dim> forcing_function;
    forcing_function.set_time(current_time + time_step);
    VectorTools::create_right_hand_side(mapping,
                                        dof_handler,
                                        QGauss<dim>(finite_element.degree + 2),
                                        forcing_function,
                                        load_vector);
    load_vector *= time_step;
    unconstrained_mass_matrix.vmult_add(load_vector, previous_solution);
  }

  // Similarly, at this point we can actually set up the correct constraints for
  // imposing the correct boundary values.
  //
  AffineConstraints<double> constraints;
  // 762: configure constraints to interpolate the boundary values at the
  // correct time.
  constraints.clear();
  ExactSolution<dim> exact_solution;
  exact_solution.set_time(current_time + time_step);
  std::vector<types::boundary_id> bdry_ids = triangulation.get_boundary_ids();
  for(unsigned int i = 0; i < bdry_ids.size(); ++i)
  {
    VectorTools::interpolate_boundary_values(dof_handler,
                                             bdry_ids[i],
                                             exact_solution,
                                             constraints);
  }  
  constraints.close();

  // At this point we are done with finite element stuff and are working at a
  // purely algebraic level. We want to set up the RHS
  //
  //     C^T (b - A k)
  //
  // where b is the unconstrained system RHS, A is the unconstrained system
  // matrix, and k is a vector containing all the inhomogeneities of the linear
  // system. The PackedOperation object constrained_right_hand_side does all of
  // this for us.
  //
  // This code looks a little weird - the intended use case for
  // constrained_right_hand_side is to use the same load vector to set up
  // multiple system right hand sides, which we don't do here.
  //
  // Finally, deal.II supports chaining together multiple matrices and
  // matrix-like operators by wrapping them with a LinearOperator class. We will
  // use this to implement our own Stokes solver later in the course. Here we
  // wrap unconstrained_system_matrix as a LinearOperator and then pass that to
  // setup_constrained_rhs, which only needs the action of the linear operator
  // and not access to matrix entries.
  auto u_system_operator = linear_operator(unconstrained_system_matrix);
  auto setup_constrained_rhs = constrained_right_hand_side(
      constraints, u_system_operator, load_vector);
  Vector<double> rhs(dof_handler.n_dofs());
  setup_constrained_rhs.apply(rhs);

  // 762: Set up the solver and preconditioner here. Based on what was done
  // above you should use *rhs* as the final right-hand side and the
  // *constrained_system_matrix* as the final matrix. Be sure to distribute
  // constraints once you are done.
  SolverControl            solver_control(1000, 1e-12);
  SolverCG<Vector<double>> solver(solver_control);

  PreconditionSSOR<SparseMatrix<double>> preconditioner;
  preconditioner.initialize(constrained_system_matrix, 1.2);

  solver.solve(constrained_system_matrix, current_solution, rhs, preconditioner);

  constraints.distribute(current_solution);
  current_time += time_step;
}

// output solution files
template <int dim>
void
Heat<dim>::output_results() const
{
  // Compute the pointwise maximum error:
  Vector<double> max_error_per_cell(triangulation.n_active_cells());
  {
    ExactSolution<dim> exact_solution;
    exact_solution.set_time(current_time);

    MappingQGeneric<dim> mapping(1);
    VectorTools::integrate_difference(mapping,
                                      dof_handler,
                                      current_solution,
                                      exact_solution,
                                      max_error_per_cell,
                                      QIterated<dim>(QGauss<1>(2), 2),
                                      VectorTools::NormType::Linfty_norm);
    std::cout << "maximum error = "
              << *std::max_element(max_error_per_cell.begin(),
                                   max_error_per_cell.end())
              << std::endl;
  }

  // Save the output:
  {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(current_solution, "solution");
    data_out.add_data_vector(max_error_per_cell, "max_error_per_cell");
    data_out.build_patches();

    std::ofstream output("output-" + std::to_string(dim) + "d-" +
                         std::to_string(n_global_refinements) + ".vtu");
    data_out.write_vtu(output);
  }
}

// run everything
template <int dim>
void
Heat<dim>::run()
{
  setup_grid();
  std::cout << "   Number of active cells:       "
            << triangulation.n_active_cells() << std::endl;

  setup_system();
  std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
            << '\n'
            << "   Timestep size: "
            << time_step
            << std::endl;

  std::cout << "   Maximal cell diameter: " << GridTools::maximal_cell_diameter(triangulation)
            << std::endl;

  assemble_system();
  do_one_time_step();
  output_results();
}


int
main()
{
  for (unsigned int i = 0; i < 9; ++i)
    {
      Heat<2> heat(i);
      heat.run();
    }

  for (unsigned int i = 0; i < 5; ++i)
    {
      Heat<3> heat(i);
      heat.run();
    }

  return 0;
}
