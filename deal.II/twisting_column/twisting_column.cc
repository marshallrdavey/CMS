#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/parameter_handler.h>
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
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/physics/elasticity/kinematics.h>

#include <math.h>

#include <filesystem>
#include <fstream>

using namespace dealii;

// Define PETSc namespace for brevity
namespace la
{
  using namespace LinearAlgebraPETSc;
}

// Define requisite parameters
void
declare_parameters(ParameterHandler &prm)
{
  prm.declare_entry("ALPHA",
                    "4.0e6",
                    Patterns::Double(),
                    "Modified neo-Hookean prameter.");
  prm.declare_entry("KAPPA", "0.0", Patterns::Double(), "Bulk modulus.");
  prm.declare_entry("RHO", "1.0", Patterns::Double(), "Density.");
  prm.declare_entry("BETA",
                    "0.25",
                    Patterns::Double(),
                    "Newmark-beta beta parameter.");
  prm.declare_entry("GAMMA",
                    "0.5",
                    Patterns::Double(),
                    "Newmark-beta gamma parameter.");
  prm.declare_entry("DT", "1.0e-5", Patterns::Double(), "Time step size.");
  prm.declare_entry("END_TIME", "1.0", Patterns::Double(), "End time.");
  prm.declare_entry("ENERGY_FREQUENCY",
                    "1",
                    Patterns::Integer(),
                    "Frequency of energy writing.");
  prm.declare_entry("OUTPUT_FREQUENCY",
                    "1",
                    Patterns::Integer(),
                    "Frequency of output writing.");
  prm.declare_entry("OUTPUT_DIRECTORY",
                    ".",
                    Patterns::Anything(),
                    "Output file path.");
}

// Generate beam
// 2x2x12 (cm)
// Cell size: 0.25^3 (cm)
template <int dim>
void
make_triangulation(parallel::shared::Triangulation<dim> &triangulation,
                   const int                             n_global_refinements)
{
  const Point<3>                  c0(-1, -1, 0);
  const Point<3>                  c1(1, 1, 12);
  const std::vector<unsigned int> rep{2, 2, 12};
  GridGenerator::subdivided_hyper_rectangle(triangulation, rep, c0, c1, true);
  triangulation.refine_global(n_global_refinements);
  return;
}

// Kronecker delta
inline double
delta(const unsigned int i, unsigned int j)
{
  return i == j ? 1.0 : 0.0;
}

// I_1 claculator, or tr(CC)
template <int dim>
inline double
I_1(const Tensor<2, dim> &FF)
{
  return trace(Physics::Elasticity::Kinematics::C(FF));
}

// Energy density function, W_DEV + W_DIL
// W_DEV: Modified neoHookean model, alpha / 2 * (I1_bar - 3)
// W_DIL: kappa / 2 * (J - 1)^2
inline double
work(const double I1, const double J, const double alpha, const double kappa)
{
  return 0.5 * alpha * (I1 / cbrt(J * J) - 3.0) +
         0.5 * kappa * (J - 1.0) * (J - 1.0);
}

// PK1_stress from the deformation gradient, dW/dFF
template <int dim>
inline Tensor<2, dim>
PK1_stress(const Tensor<2, dim> &FF,
           const double          I1,
           const double          J,
           const double          alpha,
           const double          kappa)
{
  return alpha / cbrt(J * J) * (FF - I1 * transpose(invert(FF)) / 3.0) +
         kappa * (J * J - J) * transpose(invert(FF));
}

// Jacobian of the PK1 stress, d^2W/dFF^2
template <int dim>
Tensor<4, dim>
script_A(const Tensor<2, dim> &FF,
         const double          I1,
         const double          J,
         const double          alpha,
         const double          kappa)
{
  Tensor<4, dim>       A;
  const Tensor<2, dim> FF_inv = invert(FF);
  for (unsigned int i = 0; i < dim; ++i)
    for (unsigned int j = 0; j < dim; ++j)
      for (unsigned int k = 0; k < dim; ++k)
        for (unsigned int l = 0; l < dim; ++l)
          A[i][j][k][l] =
            alpha / cbrt(J * J) *
              (2.0 / 9.0 * I1 * FF_inv[j][i] * FF_inv[l][k] -
               2.0 / 3.0 * FF[i][j] * FF_inv[l][k] + delta(i, k) * delta(j, l) -
               2.0 / 3.0 * FF[k][l] * FF_inv[j][i] +
               I1 / 3.0 * FF_inv[j][k] * FF_inv[l][i]) +
            kappa * J *
              ((2.0 * J - 1.0) * FF_inv[j][i] * FF_inv[l][k] -
               (J - 1.0) * FF_inv[j][k] * FF_inv[l][i]);
  return A;
}

// Initial velocity values
template <int dim>
class InitialVelocity : public Function<dim>
{
public:
  InitialVelocity()
    : Function<dim>(dim)
  {}

  virtual double
  value(const Point<dim> &p, unsigned int component) const override
  {
    if (component == 0)
      return -1500.0 * std::sin(numbers::PI * p[2] / 12) * p[1];
    else if (component == 1)
      return 1500.0 * std::sin(numbers::PI * p[2] / 12) * p[0];
    return 0.0;
  }

  virtual void
  vector_value(const Point<dim> &p, Vector<double> &values) const override
  {
    values = Vector<double>({-1500.0 * std::sin(numbers::PI * p[2] / 12) * p[1],
                             1500.0 * std::sin(numbers::PI * p[2] / 12) * p[0],
                             0.0});
  }
};

// Implicit solver class
template <int dim>
class ImplicitBeam
{
public:
  ImplicitBeam(const parallel::shared::Triangulation<dim> &triangulation,
               const unsigned int                          fe_order,
               const ParameterHandler                     &prm);

  void
  run();

private:
  // Class functions
  void
  setup_system();
  void
  assemble_mass_matrix();
  void
  initialize_velocity();
  void
  update_force();
  void
  initialize_acceleration();
  void
  intermediate_step();
  void
  update_step();
  void
  compute_residual();
  void
  assemble_system();
  void
  solve();
  void
  output_results(const unsigned int &step);
  void
  output_energy();

  // MPI communicator
  MPI_Comm           mpi_comm;
  ConditionalOStream pcout;

  // Mesh and finite element objects
#if DEAL_II_VERSION_GTE(9, 7, 0)
  ObserverPointer<const parallel::shared::Triangulation<dim>> tria;
#else
  SmartPointer<const parallel::shared::Triangulation<dim>> tria;
#endif
  bool                                 use_simplex;
  unsigned int                         fe_order;
  std::unique_ptr<FESystem<dim>>       fe;
  std::unique_ptr<Quadrature<dim>>     quadrature_formula;
  std::unique_ptr<Quadrature<dim - 1>> quadrature_formula_face;
  DoFHandler<dim>                      dof_handler;
  IndexSet                             locally_owned_dofs;
  IndexSet                             locally_relevant_dofs;

  // Mechanics parameters
  double m_alpha;
  double m_kappa;
  double m_rho;

  // Numerical parameters
  double       m_beta;
  double       m_gamma;
  const double m_atol = 1.e-8;

  // Time parameters
  double m_dt;
  double m_end_time;
  double m_time;

  // Output parameters
  unsigned int  m_energy_frequency;
  std::ofstream m_ostrm;
  unsigned int  m_output_frequency;
  std::string   m_output_directory;

  // Constraints
  AffineConstraints<double> constraints;

  // Matrices
  la::MPI::SparseMatrix constrained_mass_matrix;
  la::MPI::SparseMatrix unconstrained_mass_matrix;
  la::MPI::SparseMatrix constrained_system_matrix;
  la::MPI::SparseMatrix unconstrained_system_matrix;

  // Solution vectors
  la::MPI::Vector displacement;
  la::MPI::Vector local_displacement;
  la::MPI::Vector displacement_tilde;

  la::MPI::Vector velocity;
  la::MPI::Vector local_velocity;
  la::MPI::Vector velocity_tilde;

  la::MPI::Vector acceleration;
  la::MPI::Vector local_acceleration;

  la::MPI::Vector force;
  la::MPI::Vector local_force;

  la::MPI::Vector newton_update;

  la::MPI::Vector residual;
  la::MPI::Vector local_residual;
  la::MPI::Vector constrained_residual;

  // Cell-wise J storage for output
  Vector<double> J_vector;
};

// Class constructor
template <int dim>
ImplicitBeam<dim>::ImplicitBeam(
  const parallel::shared::Triangulation<dim> &triangulation,
  const unsigned int                          fe_order,
  const ParameterHandler                     &prm)
#if DEAL_II_VERSION_GTE(9, 7, 0)
  : mpi_comm(triangulation.get_mpi_communicator())
#else
  : mpi_comm(triangulation.get_communicator())
#endif
  , pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_comm) == 0))
  , tria(&triangulation)
  , use_simplex(!triangulation.all_reference_cells_are_hyper_cube())
  , fe_order(fe_order)
  , dof_handler(triangulation)
  , m_alpha(prm.get_double("ALPHA"))
  , m_kappa(prm.get_double("KAPPA"))
  , m_rho(prm.get_double("RHO"))
  , m_beta(prm.get_double("BETA"))
  , m_gamma(prm.get_double("GAMMA"))
  , m_dt(prm.get_double("DT"))
  , m_end_time(prm.get_double("END_TIME"))
  , m_energy_frequency((unsigned int)prm.get_integer("ENERGY_FREQUENCY"))
  , m_output_frequency((unsigned int)prm.get_integer("OUTPUT_FREQUENCY"))
  , m_output_directory(prm.get("OUTPUT_DIRECTORY") + '/')
{
  if (use_simplex)
    {
      fe = std::make_unique<FESystem<dim>>(FE_SimplexP<dim>(fe_order), dim);
      quadrature_formula = std::make_unique<QGaussSimplex<dim>>(fe_order + 1);
      quadrature_formula_face =
        std::make_unique<QGaussSimplex<dim - 1>>(fe_order + 1);
    }
  else
    {
      fe = std::make_unique<FESystem<dim>>(FE_Q<dim>(fe_order), dim);
      quadrature_formula      = std::make_unique<QGauss<dim>>(fe_order + 1);
      quadrature_formula_face = std::make_unique<QGauss<dim - 1>>(fe_order + 1);
    }
}

// Initialize vectors, matrices, and constraints
template <int dim>
void
ImplicitBeam<dim>::setup_system()
{
  // DoF distribution
  dof_handler.distribute_dofs(*fe);
  locally_owned_dofs    = dof_handler.locally_owned_dofs();
  locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

  // solution vectors
  displacement.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_comm);
  local_displacement.reinit(locally_owned_dofs, mpi_comm);
  displacement_tilde.reinit(locally_owned_dofs, mpi_comm);

  velocity.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_comm);
  local_velocity.reinit(locally_owned_dofs, mpi_comm);
  velocity_tilde.reinit(locally_owned_dofs, mpi_comm);

  acceleration.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_comm);
  local_acceleration.reinit(locally_owned_dofs, mpi_comm);

  force.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_comm);
  local_force.reinit(locally_owned_dofs, mpi_comm);

  newton_update.reinit(locally_owned_dofs, mpi_comm);

  residual.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_comm);
  local_residual.reinit(locally_owned_dofs, mpi_comm);
  constrained_residual.reinit(locally_owned_dofs, mpi_comm);

  J_vector.reinit(tria->n_active_cells());

  // Constrain lower z boundary to 0 displacement
  constraints.clear();
#if DEAL_II_VERSION_GTE(9, 6, 0)
  constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
#else
  constraints.reinit(locally_relevant_dofs);
#endif
  VectorTools::interpolate_boundary_values(dof_handler,
                                           4,
                                           Functions::ZeroFunction<dim>(dim),
                                           constraints);
  constraints.close();

  // Constrained matrix generation
  DynamicSparsityPattern constrained_dsp(locally_relevant_dofs);
  DoFTools::make_sparsity_pattern(dof_handler,
                                  constrained_dsp,
                                  constraints,
                                  false);
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

  // Unconstrained mass matrix
  DynamicSparsityPattern unconstrained_dsp(locally_relevant_dofs);
  DoFTools::make_sparsity_pattern(dof_handler, unconstrained_dsp);
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

// Assemble the mass matrix, does not change over time
template <int dim>
void
ImplicitBeam<dim>::assemble_mass_matrix()
{
  constrained_mass_matrix   = 0;
  unconstrained_mass_matrix = 0;

  FEValues<dim> fe_values(*fe,
                          *quadrature_formula,
                          update_values | update_JxW_values);

  const unsigned int dofs_per_cell = fe->n_dofs_per_cell();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        cell_matrix = 0;

        fe_values.reinit(cell);

        for (const unsigned int q_index : fe_values.quadrature_point_indices())
          for (const unsigned int i : fe_values.dof_indices())
            {
              const unsigned int i_component =
                fe->system_to_component_index(i).first;
              for (const unsigned int j : fe_values.dof_indices())
                {
                  const unsigned int j_component =
                    fe->system_to_component_index(j).first;
                  cell_matrix(i, j) +=
                    m_rho * ((j_component == i_component) ?
                               fe_values.shape_value(i, q_index) *
                                 fe_values.shape_value(j, q_index) *
                                 fe_values.JxW(q_index) :
                               0.0);
                }
            }
        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(cell_matrix,
                                               local_dof_indices,
                                               constrained_mass_matrix);
        unconstrained_mass_matrix.add(local_dof_indices, cell_matrix);
      }
  constrained_mass_matrix.compress(VectorOperation::add);
  unconstrained_mass_matrix.compress(VectorOperation::add);
}

// Initialize the velocity
template <int dim>
void
ImplicitBeam<dim>::initialize_velocity()
{
  VectorTools::project(dof_handler,
                       constraints,
                       *quadrature_formula,
                       InitialVelocity<dim>(),
                       local_velocity);
}

// Update the force vector, compute J
template <int dim>
void
ImplicitBeam<dim>::update_force()
{
  local_force = 0;
  J_vector    = 0;

  FEValues<dim> fe_values(*fe,
                          *quadrature_formula,
                          update_values | update_gradients | update_JxW_values);

  FEValuesExtractors::Vector  u_fe(0);
  std::vector<Tensor<2, dim>> qp_Grad_u;

  const unsigned int dofs_per_cell = fe->n_dofs_per_cell();

  Vector<double> cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  double new_volume;

  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        cell_rhs   = 0;
        new_volume = 0;

        fe_values.reinit(cell);

        const unsigned int n_q_points = fe_values.get_quadrature().size();
        qp_Grad_u.resize(n_q_points);
        fe_values[u_fe].get_function_gradients(displacement, qp_Grad_u);

        for (const unsigned int q_index : fe_values.quadrature_point_indices())
          {
            const Tensor<2, dim> FF =
              Physics::Elasticity::Kinematics::F(qp_Grad_u[q_index]);

            const double I1 = I_1(FF);

            const double J = determinant(FF);
            new_volume += J * fe_values.JxW(q_index);

            const Tensor<2, dim> PP = PK1_stress(FF, I1, J, m_alpha, m_kappa);

            for (const unsigned int i : fe_values.dof_indices())
              {
                const unsigned int i_component =
                  fe->system_to_component_index(i).first;
                for (unsigned int k = 0; k < dim; ++k)
                  {
                    cell_rhs(i) -= PP[i_component][k] *
                                   fe_values.shape_grad(i, q_index)[k] *
                                   fe_values.JxW(q_index);
                  }
              }
          }
        cell->get_dof_indices(local_dof_indices);
        local_force.add(local_dof_indices, cell_rhs);
        J_vector[cell->active_cell_index()] = new_volume / cell->measure();
      }
  local_force.compress(VectorOperation::add);
  force = local_force;
}

// Initialize acceleration, M a = f
template <int dim>
void
ImplicitBeam<dim>::initialize_acceleration()
{
  // Sovler settings
  SolverControl solver_control(10000, 1e-16);
#if DEAL_II_VERSION_GTE(9, 5, 0)
  la::SolverCG solver(solver_control);
#else
  la::SolverCG solver(solver_control, mpi_comm);
#endif

  // Preconditioner settings
  la::MPI::PreconditionAMG                 preconditioner;
  la::MPI::PreconditionAMG::AdditionalData data;
  data.symmetric_operator = true;
  preconditioner.initialize(constrained_mass_matrix, data);

  // Constrain the force
  auto force_system_operator =
    linear_operator<la::MPI::Vector>(unconstrained_mass_matrix);
  auto setup_constrained_force =
    constrained_right_hand_side<la::MPI::Vector>(constraints,
                                                 force_system_operator,
                                                 force);
  la::MPI::Vector force_rhs(locally_owned_dofs, mpi_comm);
  setup_constrained_force.apply(force_rhs);

  // Solve
  solver.solve(constrained_mass_matrix,
               local_acceleration,
               force_rhs,
               preconditioner);
  constraints.distribute(local_acceleration);
}

// Compute velocity_tilde and displacement_tilde
template <int dim>
void
ImplicitBeam<dim>::intermediate_step()
{
  // Intermediate displacement
  displacement_tilde = 0;
  displacement_tilde.add(1.0, local_displacement);
  displacement_tilde.add(m_dt, local_velocity);
  displacement_tilde.add(m_dt * m_dt * (1.0 - 2.0 * m_beta) / 2.0,
                         local_acceleration);

  // Intermediate velocity
  velocity_tilde = 0;
  velocity_tilde.add(1.0, local_velocity);
  velocity_tilde.add(m_dt * (1.0 - m_gamma), local_acceleration);
}

// Update velocity, acceleration, and the residual
template <int dim>
void
ImplicitBeam<dim>::update_step()
{
  // Update acceleration
  local_acceleration = 0;
  local_acceleration.add(1.0 / m_beta / m_dt / m_dt, local_displacement);
  local_acceleration.add(-1.0 / m_beta / m_dt / m_dt, displacement_tilde);

  // Update velocity
  local_velocity = 0;
  local_velocity.add(1.0, velocity_tilde);
  local_velocity.add(m_gamma * m_dt, local_acceleration);
}

template <int dim>
void
ImplicitBeam<dim>::compute_residual()
{
  // Update the residual
  local_residual = 0;
  local_residual.add(1.0, local_force);
  const la::MPI::Vector negative_acceleration(-1.0 * local_acceleration);
  unconstrained_mass_matrix.vmult_add(local_residual, negative_acceleration);

  // Constrain the residual
  constrained_residual = 0;
  residual             = local_residual;
  auto residual_system_operator =
    linear_operator<la::MPI::Vector>(unconstrained_system_matrix);
  auto setup_constrained_residual =
    constrained_right_hand_side<la::MPI::Vector>(constraints,
                                                 residual_system_operator,
                                                 residual);
  setup_constrained_residual.apply(constrained_residual);
}

// Assembly for Newton solves
template <int dim>
void
ImplicitBeam<dim>::assemble_system()
{
  constrained_system_matrix   = 0;
  unconstrained_system_matrix = 0;

  FEValues<dim> fe_values(*fe,
                          *quadrature_formula,
                          update_values | update_gradients | update_JxW_values);

  FEValuesExtractors::Vector  u_fe(0);
  std::vector<Tensor<2, dim>> qp_Grad_u;

  const unsigned int dofs_per_cell = fe->n_dofs_per_cell();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        cell_matrix = 0;

        fe_values.reinit(cell);

        const unsigned int n_q_points = fe_values.get_quadrature().size();
        qp_Grad_u.resize(n_q_points);
        fe_values[u_fe].get_function_gradients(displacement, qp_Grad_u);

        for (const unsigned int q_index : fe_values.quadrature_point_indices())
          {
            const Tensor<2, dim> FF =
              Physics::Elasticity::Kinematics::F(qp_Grad_u[q_index]);

            const double I1 = I_1(FF);

            const double J = determinant(FF);

            const Tensor<4, dim> AA = script_A(FF, I1, J, m_alpha, m_kappa);

            for (const unsigned int i : fe_values.dof_indices())
              {
                const unsigned int i_component =
                  fe->system_to_component_index(i).first;
                for (const unsigned int j : fe_values.dof_indices())
                  {
                    const unsigned int j_component =
                      fe->system_to_component_index(j).first;

                    cell_matrix(i, j) +=
                      m_rho / m_beta / m_dt / m_dt *
                      ((j_component == i_component) ?
                         fe_values.shape_value(i, q_index) *
                           fe_values.shape_value(j, q_index) *
                           fe_values.JxW(q_index) :
                         0.0);
                    for (unsigned int k = 0; k < dim; ++k)
                      for (unsigned int l = 0; l < dim; ++l)
                        {
                          cell_matrix(i, j) +=
                            AA[i_component][k][j_component][l] *
                            fe_values.shape_grad(i, q_index)[k] *
                            fe_values.shape_grad(j, q_index)[l] *
                            fe_values.JxW(q_index);
                        }
                  }
              }
          }
        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(cell_matrix,
                                               local_dof_indices,
                                               constrained_system_matrix);
        unconstrained_system_matrix.add(local_dof_indices, cell_matrix);
      }
  constrained_system_matrix.compress(VectorOperation::add);
  unconstrained_system_matrix.compress(VectorOperation::add);
}

// Newton solve
template <int dim>
void
ImplicitBeam<dim>::solve()
{
  // Intialize solver
  SolverControl solver_control(10000, 1.0e-16);
#if DEAL_II_VERSION_GTE(9, 5, 0)
  la::SolverGMRES solver(solver_control);
#else
  la::SolverGMRES solver(solver_control, mpi_comm);
#endif

  // Preconditioner settings
  la::MPI::PreconditionJacobi                 preconditioner;
  la::MPI::PreconditionJacobi::AdditionalData data;
  preconditioner.initialize(constrained_system_matrix, data);

  // Solve for newton update
  newton_update = 0;
  solver.solve(constrained_system_matrix,
               newton_update,
               constrained_residual,
               preconditioner);

  // Constrain newton update
  constraints.distribute(newton_update);
}

// Ouptut function
template <int dim>
void
ImplicitBeam<dim>::output_results(const unsigned int &step)
{
  // Build data out object
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);

  // Define displacement names
  std::vector<std::string> displacement_names;
  displacement_names.emplace_back("x_displacement");
  displacement_names.emplace_back("y_displacement");
  displacement_names.emplace_back("z_displacement");

  // Define velocity names
  std::vector<std::string> velocity_names;
  velocity_names.emplace_back("x_velocity");
  velocity_names.emplace_back("y_velocity");
  velocity_names.emplace_back("z_velocity");

  // Define acceleration names
  std::vector<std::string> acceleration_names;
  acceleration_names.emplace_back("x_acceleration");
  acceleration_names.emplace_back("y_acceleration");
  acceleration_names.emplace_back("z_acceleration");

  // Update velocity and acceleration to ghosted vectors
  velocity     = local_velocity;
  acceleration = local_acceleration;

  // Add data to data out object
  data_out.add_data_vector(displacement, displacement_names);
  data_out.add_data_vector(velocity, velocity_names);
  data_out.add_data_vector(acceleration, acceleration_names);
  data_out.add_data_vector(J_vector, "J");

  // Correlate time to time step
  data_out.set_flags(DataOutBase::VtkFlags(m_time, step));

  // Build patches and write in parallel
  data_out.build_patches();
  if (use_simplex)
    data_out.write_vtu_with_pvtu_record(
      "./simplex_output/", "solution", step, mpi_comm, 4);
  else
    data_out.write_vtu_with_pvtu_record(
      m_output_directory, "solution", step, mpi_comm, 4);
}

template <int dim>
void
ImplicitBeam<dim>::output_energy()
{
  displacement = local_displacement;
  velocity     = local_velocity;

  FEValues<dim> fe_values(*fe,
                          *quadrature_formula,
                          update_values | update_gradients | update_JxW_values);

  FEValuesExtractors::Vector  qp_fe(0);
  std::vector<Tensor<1, dim>> qp_v;
  std::vector<Tensor<2, dim>> qp_Grad_u;

  double local_kinetic_energy = 0.0;
  double local_elastic_energy = 0.0;

  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        fe_values.reinit(cell);

        const unsigned int n_q_points = fe_values.get_quadrature().size();

        qp_v.resize(n_q_points);
        qp_Grad_u.resize(n_q_points);

        fe_values[qp_fe].get_function_values(velocity, qp_v);
        fe_values[qp_fe].get_function_gradients(displacement, qp_Grad_u);

        for (const unsigned int q_index : fe_values.quadrature_point_indices())
          {
            const Tensor<2, dim> FF =
              Physics::Elasticity::Kinematics::F(qp_Grad_u[q_index]);
            const double I1 = I_1(FF);
            const double J  = determinant(FF);

            local_kinetic_energy += 0.5 * m_rho * qp_v[q_index].norm_square() *
                                    fe_values.JxW(q_index);
            local_elastic_energy +=
              work(I1, J, m_alpha, m_kappa) * fe_values.JxW(q_index);
          }
      }

  double total_kinetic_energy = 0.0;
  double total_elastic_energy = 0.0;
  MPI_Reduce(&local_kinetic_energy,
             &total_kinetic_energy,
             1,
             MPI_DOUBLE,
             MPI_SUM,
             0,
             mpi_comm);
  MPI_Reduce(&local_elastic_energy,
             &total_elastic_energy,
             1,
             MPI_DOUBLE,
             MPI_SUM,
             0,
             mpi_comm);
  if (Utilities::MPI::this_mpi_process(mpi_comm) == 0)
    {
      m_ostrm.open(m_output_directory + "energy.dat",
                   std::ios::out | std::ios::app);
      m_ostrm << std::hexfloat << m_time << ' ' << total_kinetic_energy << ' '
              << total_elastic_energy << ' '
              << total_elastic_energy + total_kinetic_energy << '\n';
      m_ostrm.close();
    }
}

// Run function
template <int dim>
void
ImplicitBeam<dim>::run()
{
  pcout << " Number of active cells:       " << tria->n_active_cells() << '\n';
  setup_system();
  pcout << " Number of degrees of freedom: " << dof_handler.n_dofs() << '\n';
  pcout << " Maximal cell diameter: " << GridTools::maximal_cell_diameter(*tria)
        << "\n\n";
  pcout << " Alpha: " << m_alpha << '\n';
  pcout << " Kappa: " << m_kappa << '\n';
  pcout << " DT: " << m_dt << "\n\n";


  assemble_mass_matrix();

  // Initialize problem
  m_time                = 0.0;
  unsigned int step     = 0;
  unsigned int out_step = 0;
  initialize_velocity();
  local_displacement = 0;
  output_energy();

  // Initialize acceleration
  update_force();
  initialize_acceleration();
  output_results(out_step);

  // Main time loop
  while (m_time < m_end_time - m_dt / 2.)
    {
      ++step;
      m_time += m_dt;

      intermediate_step();
      update_step();
      compute_residual();

      pcout << " Time = " << m_time << "\n";
      pcout << "   Initial residual at time step " << step << ": "
            << constrained_residual.l2_norm() << "\n";

      unsigned int count   = 0;
      double       rel_tol = 100;

      while (count < 20 && constrained_residual.l2_norm() > m_atol &&
             rel_tol > 1.09)
        {
          rel_tol = constrained_residual.l2_norm();
          assemble_system();
          solve();
          local_displacement.add(1., newton_update);
          displacement = local_displacement;

          update_force();
          update_step();
          compute_residual();

          ++count;
          rel_tol /= constrained_residual.l2_norm();
#ifndef NO_NEWTON_RESIDUALS
          pcout << "     Residual after newton step " << count << ": "
                << constrained_residual.l2_norm() << " | " << rel_tol << "\n";
#endif
        }

      pcout << "   Final residual after " << count
            << " newton iterations: " << constrained_residual.l2_norm()
            << "\n\n";

      if (step % m_output_frequency == 0)
        output_results(++out_step);
      if (step % m_energy_frequency == 0)
        output_energy();
    }
}

int
main(int argc, char **argv)
{
  // set up MPI communicator
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  MPI_Comm                         mpi_communicator = MPI_COMM_WORLD;

  // Set up input parameters
  ParameterHandler prm;
  declare_parameters(prm);

  // Allow for users to pass in an input file from the command line. Default to
  // a local file named twisting_column.prm.
  if (argc > 1)
    prm.parse_input(argv[1]);
  else
    prm.parse_input("twisting_column.prm");

  // Create the output directory from the OUTPUT_DIRECTORY parameter.
  try
    {
      std::filesystem::create_directories(prm.get("OUTPUT_DIRECTORY"));
    }
  catch (const std::filesystem::filesystem_error &e)
    {
      std::cerr << "Directory creation error: " << e.what() << '\n';
    }

  // Make triangulation
  const int                          n_global_refinements = 2;
  parallel::shared::Triangulation<3> triangulation(mpi_communicator);
  make_triangulation(triangulation, n_global_refinements);

  // Run the model using hexes
  ImplicitBeam<3> implicit_test(triangulation, 2, prm);
  implicit_test.run();

  return 0;
}
