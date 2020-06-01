package us.ihmc.convexOptimization.quadraticProgram;

import org.ejml.data.DenseMatrix64F;

import us.ihmc.convexOptimization.exceptions.NoConvergenceException;

public interface ActiveSetQPSolver
{
   void setConvergenceThreshold(double convergenceThreshold);

   void setMaxNumberOfIterations(int maxNumberOfIterations);

   void clear();

   void setLowerBounds(DenseMatrix64F variableLowerBounds);

   void setUpperBounds(DenseMatrix64F variableUpperBounds);

   default void setVariableBounds(DenseMatrix64F variableLowerBounds, DenseMatrix64F variableUpperBounds)
   {
      setLowerBounds(variableLowerBounds);
      setUpperBounds(variableUpperBounds);
   }

   void setQuadraticCostFunction(DenseMatrix64F costQuadraticMatrix, DenseMatrix64F costLinearVector, double quadraticCostScalar);

   double getObjectiveCost(DenseMatrix64F x);

   void setLinearEqualityConstraints(DenseMatrix64F linearEqualityConstraintsAMatrix, DenseMatrix64F linearEqualityConstraintsBVector);

   void setLinearInequalityConstraints(DenseMatrix64F linearInequalityConstraintCMatrix, DenseMatrix64F linearInequalityConstraintDVector);

   int solve(double[] solutionToPack) throws NoConvergenceException;

   int solve(DenseMatrix64F solutionToPack) throws NoConvergenceException;

}