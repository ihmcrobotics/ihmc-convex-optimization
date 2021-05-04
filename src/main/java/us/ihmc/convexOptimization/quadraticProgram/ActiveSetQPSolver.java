package us.ihmc.convexOptimization.quadraticProgram;

import org.ejml.data.DMatrix;
import org.ejml.data.DMatrixRMaj;

/**
 * General interface for a QP solver using under the hood an active set approach, see:
 * <ul>
 * <li>General reference for quadratic programming:
 * <a href="https://en.wikipedia.org/wiki/Quadratic_programming">Wikipedia QP</a>.
 * <li>Active-set method: <a href="https://en.wikipedia.org/wiki/Active-set_method">Wikipedia
 * Active-set</a>.
 * </ul>
 * <p>
 * Problem formulation:
 * 
 * <pre>
 * min f(x) = 0.5 * x<sup>T</sup> H x + f<sup>T</sup>x + c
 * s.t.
 *    A<sub>in</sub> x &leq; b<sub>in</sub>
 *    A<sub>eq</sub> x = b<sub>eq</sub>
 *    x &geq; x<sub>min</sub>
 *    x &leq; x<sub>max</sub>
 * </pre>
 * 
 * where:
 * <ul>
 * <li>N, M<sub>in</sub>, and M<sub>eq</sub> are the number of variables, the number of inequality
 * constraints, and the number of equality constraints, respectively.
 * <li><tt>x</tt> is the N-by-1 variable vector to find the solution for.
 * <li><tt>H</tt> is the N-by-N (typically positive semi-definite, preferably positive definite)
 * Hessian matrix.
 * <li><tt>f</tt> is the N-by-1 gradient vector.
 * <li><tt>c</tt> is a scalar constant, that is typically equal to zero.
 * <li><tt>A<sub>in</sub></tt> and <tt>b<sub>in</sub></tt> are respectively a M<sub>in</sub>-by-N
 * matrix and M<sub>in</sub>-by-1 vector used to formulate M<sub>in</sub> inequality constraints.
 * <li><tt>A<sub>eq</sub></tt> and <tt>b<sub>eq</sub></tt> are respectively a M<sub>eq</sub>-by-N
 * matrix and M<sub>eq</sub>-by-1 vector used to formulate M<sub>eq</sub> equality constraints.
 * <li><tt>x<sub>min</sub></tt> is the N-by-1 vector used for enforcing a lower bound on each
 * element of <tt>x</tt>.
 * <li><tt>x<sub>max</sub></tt> is the N-by-1 vector used for enforcing an upper bound on each
 * element of <tt>x</tt>.
 * </ul>
 * </p>
 */
public interface ActiveSetQPSolver
{
   /**
    * Threshold used for triggering changes to the active-set. Internal usage is subject to the
    * implementation of the solver.
    * 
    * @param convergenceThreshold the threshold for modifying the internal active-set when solving.
    */
   void setConvergenceThreshold(double convergenceThreshold);

   /**
    * Sets the maximum number of times the active-set can be modified before considering the problem as
    * unsolvable.
    * 
    * @param maxNumberOfIterations the upper limit on the number of iterations before failing when
    *                              solving.
    */
   void setMaxNumberOfIterations(int maxNumberOfIterations);

   /**
    * Sets whether to use a warm-start or not.
    * <p>
    * When warm-start is enabled, the last active-set from the last solve is used as an initial guess
    * for the next solve initial active-set. This cuts down considerably the number of iterations when
    * the active-set from one solve to the next is expected to not change or only change a little.
    * </p>
    * 
    * @param useWarmStart {@code true} to use warm-start, {@code false} otherwise.
    */
   void setUseWarmStart(boolean useWarmStart);

   /**
    * Indicates that for the next solve, the active-set should be reset, i.e. not start from the last
    * solve's active-set.
    * <p>
    * This method is only effective when warm-start is enabled.
    * </p>
    */
   void resetActiveSet();

   /**
    * Clears the internal data but does not reset the active-set.
    */
   void clear();

   /**
    * Sets the N-by-1 vector used to enforce a lower bound on the problem's variables:
    * 
    * <pre>
    * x &geq; x<sub>min</sub>
    * </pre>
    * 
    * where N is the number of variables in <tt>x</tt>.
    * 
    * @param xMin the lower bound for the problem's variables. Not modified.
    * @see ActiveSetQPSolver
    */
   void setLowerBounds(DMatrix xMin);

   /**
    * Sets the N-by-1 vector used to enforce an upper bound on the problem's variables:
    * 
    * <pre>
    * x &geq; x<sub>max</sub>
    * </pre>
    * 
    * where N is the number of variables in <tt>x</tt>.
    * 
    * @param xMax the upper bound for the problem's variables. Not modified.
    * @see ActiveSetQPSolver
    */
   void setUpperBounds(DMatrix xMax);

   /**
    * Sets the N-by-1 vectors used to enforce a lower and upper bounds on the problem's variables:
    * 
    * <pre>
    * x &geq; x<sub>min</sub>
    * x &geq; x<sub>max</sub>
    * </pre>
    * 
    * where N is the number of variables in <tt>x</tt>.
    * 
    * @param xMin the lower bound for the problem's variables. Not modified.
    * @param xMax the upper bound for the problem's variables. Not modified.
    * @see ActiveSetQPSolver
    */
   default void setVariableBounds(DMatrix xMin, DMatrix xMax)
   {
      setLowerBounds(xMin);
      setUpperBounds(xMax);
   }

   /**
    * Configures the function to minimize:
    * 
    * <pre>
    * f(x) = 0.5 * x<sup>T</sup> H x + f<sup>T</sup>x
    * </pre>
    * 
    * where <tt>x</tt> is the N-by-1 vector of variables to solve for.
    * 
    * @param H is the N-by-N (typically positive semi-definite, preferably positive definite) Hessian
    *          matrix. Not modified.
    * @param f is the N-by-1 gradient vector.
    * @see ActiveSetQPSolver
    */
   default void setQuadraticCostFunction(DMatrix H, DMatrix f)
   {
      setQuadraticCostFunction(H, f, 0.0);
   }

   /**
    * Configures the function to minimize:
    * 
    * <pre>
    * f(x) = 0.5 * x<sup>T</sup> H x + f<sup>T</sup>x + c
    * </pre>
    * 
    * where <tt>x</tt> is the N-by-1 vector of variables to solve for.
    * 
    * @param H is the N-by-N (typically positive semi-definite, preferably positive definite) Hessian
    *          matrix. Not modified.
    * @param f is the N-by-1 gradient vector. Not modified.
    * @param c is a scalar constant, that is typically equal to zero.
    * @see ActiveSetQPSolver
    */
   void setQuadraticCostFunction(DMatrix H, DMatrix f, double c);

   /**
    * Configures the linear equality constraints:
    * 
    * <pre>
    * A<sub>eq</sub> x = b<sub>eq</sub>
    * </pre>
    * 
    * where:
    * <ul>
    * <li><tt>x</tt> is the N-by-1 vector of variables to solve for.
    * <li>M<sub>eq</sub> is the number of equality constraints.
    * </ul>
    * 
    * @param Aeq is a M<sub>eq</sub>-by-N matrix. Not modified.
    * @param beq is a M<sub>eq</sub>-by-1 vector. Not modified.
    * @see ActiveSetQPSolver
    */
   void setLinearEqualityConstraints(DMatrix Aeq, DMatrix beq);

   /**
    * Configures the linear inequality constraints:
    * 
    * <pre>
    * A<sub>in</sub> x &leq; b<sub>in</sub>
    * </pre>
    * 
    * where:
    * <ul>
    * <li><tt>x</tt> is the N-by-1 vector of variables to solve for.
    * <li>M<sub>in</sub> is the number of inequality constraints.
    * </ul>
    * 
    * @param Ain is a M<sub>in</sub>-by-N matrix. Not modified.
    * @param bin is a M<sub>in</sub>-by-1 vector. Not modified.
    * @see ActiveSetQPSolver
    */
   void setLinearInequalityConstraints(DMatrix Ain, DMatrix bin);

   /**
    * With the problem previously formulated, solves the objective function for {@code x}:
    * 
    * <pre>
    * min f(x) = 0.5 * x<sup>T</sup> H x + f<sup>T</sup>x + c
    * s.t.
    *    A<sub>in</sub> x &leq; b<sub>in</sub>
    *    A<sub>eq</sub> x = b<sub>eq</sub>
    *    x &geq; x<sub>min</sub>
    *    x &leq; x<sub>max</sub>
    * </pre>
    * 
    * where:
    * <ul>
    * <li>N, M<sub>in</sub>, and M<sub>eq</sub> are the number of variables, the number of inequality
    * constraints, and the number of equality constraints, respectively.
    * <li><tt>x</tt> is the N-by-1 variable vector to find the solution for.
    * <li><tt>H</tt> is the N-by-N (typically positive semi-definite, preferably positive definite)
    * Hessian matrix.
    * <li><tt>f</tt> is the N-by-1 gradient vector.
    * <li><tt>c</tt> is a scalar constant, that is typically equal to zero.
    * <li><tt>A<sub>in</sub></tt> and <tt>b<sub>in</sub></tt> are respectively a M<sub>in</sub>-by-N
    * matrix and M<sub>in</sub>-by-1 vector used to formulate M<sub>in</sub> inequality constraints.
    * <li><tt>A<sub>eq</sub></tt> and <tt>b<sub>eq</sub></tt> are respectively a M<sub>eq</sub>-by-N
    * matrix and M<sub>eq</sub>-by-1 vector used to formulate M<sub>eq</sub> equality constraints.
    * <li><tt>x<sub>min</sub></tt> is the N-by-1 vector used for enforcing a lower bound on each
    * element of <tt>x</tt>.
    * <li><tt>x<sub>max</sub></tt> is the N-by-1 vector used for enforcing an upper bound on each
    * element of <tt>x</tt>.
    * </ul>
    * <p>
    * If this method fails, {@code xToPack} is filled with {@link Double#NaN}.
    * </p>
    * 
    * @param xToPack the N-by-1 vector in which the solution is stored, or filled with
    *                {@link Double#NaN} if this method failed, i.e. the maximum number of iteration has
    *                been reached before finding a solution to the problem. Modified.
    * @return the number of iterations it took to find the solution. An iteration is defined as a
    *         change in the active-set.
    * @see #setQuadraticCostFunction(DMatrix, DMatrix)
    * @see #setLinearInequalityConstraints(DMatrix, DMatrix)
    * @see #setLinearEqualityConstraints(DMatrix, DMatrix)
    * @see #setLowerBounds(DMatrix)
    * @see #setUpperBounds(DMatrix)
    * @see ActiveSetQPSolver
    */
   int solve(DMatrixRMaj xToPack);

   /**
    * Calculates the cost from the objective function given value for {@code x}:
    * 
    * <pre>
    * f(x) = 0.5 * x<sup>T</sup> H x + f<sup>T</sup>x
    * </pre>
    * 
    * @param x is a N-by-1 vector. Not modified.
    * @return the cost value.
    */
   double getObjectiveCost(DMatrixRMaj x);

   /**
    * <b>This is for advanced users or testing only.</b>
    * <p>
    * Packs the Lagrange multipliers corresponding to the linear equality constraints.
    * </p>
    * 
    * @param multipliersMatrixToPack the M<sub>eq</sub>-by-1 vector in which the Lagrange multipliers
    *                                resulting from the last solve are stored, where M<sub>eq</sub> is
    *                                the number of equality constraints. Modified.
    */
   void getLagrangeEqualityConstraintMultipliers(DMatrixRMaj multipliersMatrixToPack);

   /**
    * <b>This is for advanced users or testing only.</b>
    * <p>
    * Packs the Lagrange multipliers corresponding to the linear inequality constraints.
    * </p>
    * 
    * @param multipliersMatrixToPack the M*<sub>in</sub>-by-1 vector in which the Lagrange multipliers
    *                                resulting from the last solve are stored, where M*<sub>in</sub> is
    *                                the number of active inequality constraints. Modified.
    */
   void getLagrangeInequalityConstraintMultipliers(DMatrixRMaj multipliersMatrixToPack);

   /**
    * <b>This is for advanced users or testing only.</b>
    * <p>
    * Packs the Lagrange multipliers corresponding to the lower bound constraints.
    * </p>
    * 
    * @param multipliersMatrixToPack the N*-by-1 vector in which the Lagrange multipliers resulting
    *                                from the last solve are stored, where N* is the number of active
    *                                lower bound constraints. Modified.
    */
   void getLagrangeLowerBoundsMultipliers(DMatrixRMaj multipliersMatrixToPack);

   /**
    * <b>This is for advanced users or testing only.</b>
    * <p>
    * Packs the Lagrange multipliers corresponding to the upper bound constraints.
    * </p>
    * 
    * @param multipliersMatrixToPack the N*-by-1 vector in which the Lagrange multipliers resulting
    *                                from the last solve are stored, where N* is the number of active
    *                                upper bound constraints. Modified.
    */
   void getLagrangeUpperBoundsMultipliers(DMatrixRMaj multipliersMatrixToPack);
}