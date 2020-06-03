package us.ihmc.convexOptimization.quadraticProgram;

import org.ejml.data.DMatrixRMaj;

public class SimpleActiveSetQPSolver extends AbstractActiveSetQPSolver
{
   // Uses the algorithm and naming convention found in MIT Paper
   // "An efficiently solvable quadratic program for stabilizing dynamic locomotion"
   // by Scott Kuindersma, Frank Permenter, and Russ Tedrakenv.

   SimpleActiveSetQPStandaloneSolver solver = new SimpleActiveSetQPStandaloneSolver();
   DMatrixRMaj solutionVector = new DMatrixRMaj(0);

   @Override
   public double[] solve()
   {
      int iterations = solver.solve(quadraticCostGMatrix,
                                    quadraticCostFVector,
                                    linearEqualityConstraintA,
                                    linearEqualityConstraintB,
                                    linearInequalityConstraintA,
                                    linearInequalityConstraintB,
                                    linearInequalityActiveSet,
                                    solutionVector);
      return solutionVector.getData();
   }
}
