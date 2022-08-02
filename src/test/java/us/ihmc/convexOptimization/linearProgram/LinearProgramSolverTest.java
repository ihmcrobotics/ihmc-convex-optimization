package us.ihmc.convexOptimization.linearProgram;

import org.apache.commons.math3.optim.MaxIter;
import org.apache.commons.math3.optim.linear.*;
import org.apache.commons.math3.optim.nonlinear.scalar.GoalType;
import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import us.ihmc.commons.MathTools;
import us.ihmc.euclid.tools.EuclidCoreRandomTools;
import us.ihmc.euclid.tools.EuclidCoreTools;
import us.ihmc.matrixlib.MatrixTools;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class LinearProgramSolverTest
{
   private static final Random random = new Random(349034);
   private static final double epsilon = 1e-5;

   private static class ConstraintSet
   {
      private final DMatrixRMaj inequalityMatrix = new DMatrixRMaj(0);
      private final DMatrixRMaj inequalityVector = new DMatrixRMaj(0);
      private final DMatrixRMaj equalityMatrix = new DMatrixRMaj(0);
      private final DMatrixRMaj equalityVector = new DMatrixRMaj(0);
   }

   @Test
   public void testOnlyInequalityConstraints_MaxBounded()
   {
      int tests = 400;
      int costVectorsPerProblem = 10;

      for (int i = 0; i < tests; i++)
      {
         ConstraintSet constraintSet = generateRandomEllipsoidBasedConstraintSet(false, false);
         runTest(constraintSet, costVectorsPerProblem);
      }
   }

   @Test
   public void testOnlyInequalityConstraints_MinBounded()
   {
      int tests = 400;
      int costVectorsPerProblem = 10;

      for (int i = 0; i < tests; i++)
      {
         ConstraintSet constraintSet = generateRandomEllipsoidBasedConstraintSet(false, false);
         DMatrixRMaj A = constraintSet.inequalityMatrix;
         DMatrixRMaj b = constraintSet.inequalityVector;
         CommonOps_DDRM.scale(-1.0, A);
         CommonOps_DDRM.scale(-1.0, b);

         runTest(constraintSet, costVectorsPerProblem);
      }
   }

   @Test
   public void testWithEqualityConstraintsInInequalityMatrix()
   {
      int tests = 100;
      int costVectorsPerProblem = 10;

      for (int i = 0; i < tests; i++)
      {
         ConstraintSet constraintSet = generateRandomEllipsoidBasedConstraintSet(true, true);
         runTest(constraintSet, costVectorsPerProblem);
      }
   }

   @Test
   public void testWithEqualityConstraintsGivenExplicitly()
   {
      int tests = 100;
      int costVectorsPerProblem = 10;

      for (int i = 0; i < tests; i++)
      {
         ConstraintSet constraintSet = generateRandomEllipsoidBasedConstraintSet(true, false);
         runTest(constraintSet, costVectorsPerProblem);
      }
   }

   @Test
   public void testRandomLPs()
   {
      int tests = 200;
      int costVectorsPerProblem = 10;

      for (int i = 0; i < tests; i++)
      {
         ConstraintSet constraintSet = generateRandomConstraints();
         runTest(constraintSet, costVectorsPerProblem);
      }
   }

   private static void runTest(ConstraintSet constraintSet, int numberOfTests)
   {
      LinearProgramSolver customSolver = new LinearProgramSolver();

      for (int i = 0; i < numberOfTests; i++)
      {
         DMatrixRMaj costVector = generateRandomCostVector(constraintSet.inequalityMatrix.getNumCols());

         // SOLVE WITH APACHE //
         double[] apacheCommonsSolution = solveWithApacheCommons(constraintSet, costVector, Relationship.LEQ);

         DMatrixRMaj simplexSolution = new DMatrixRMaj(0);
         DMatrixRMaj crissCrossSolution = new DMatrixRMaj(0);

         boolean foundSimplexSolution, foundCrissCrossSolution;
         if (constraintSet.equalityMatrix.getNumRows() > 0)
         {
            foundSimplexSolution = customSolver.solve(costVector, constraintSet.inequalityMatrix, constraintSet.inequalityVector, constraintSet.equalityMatrix, constraintSet.equalityVector, simplexSolution, SolverMethod.SIMPLEX);
            foundCrissCrossSolution = customSolver.solve(costVector, constraintSet.inequalityMatrix, constraintSet.inequalityVector, constraintSet.equalityMatrix, constraintSet.equalityVector, crissCrossSolution, SolverMethod.CRISS_CROSS);
         }
         else
         {
            foundSimplexSolution = customSolver.solve(costVector, constraintSet.inequalityMatrix, constraintSet.inequalityVector, simplexSolution, SolverMethod.SIMPLEX);
            foundCrissCrossSolution = customSolver.solve(costVector, constraintSet.inequalityMatrix, constraintSet.inequalityVector, crissCrossSolution, SolverMethod.CRISS_CROSS);
         }

         if (apacheCommonsSolution == null)
         {
            Assertions.assertFalse(foundSimplexSolution);
            Assertions.assertFalse(foundCrissCrossSolution);
         }
         else
         {
            Assertions.assertTrue(foundSimplexSolution);
            Assertions.assertTrue(foundCrissCrossSolution);

            for (int k = 0; k < apacheCommonsSolution.length; k++)
            {
               Assertions.assertTrue(EuclidCoreTools.epsilonEquals(apacheCommonsSolution[k], simplexSolution.get(k), epsilon));
               Assertions.assertTrue(EuclidCoreTools.epsilonEquals(apacheCommonsSolution[k], crissCrossSolution.get(k), epsilon));
            }
         }
      }
   }

   /**
    * Sets inequality matrices to be planes that are tangent to an ellipsoid. Sets equality matrices to converge at a point interior to the ellipsoid.
    */
   private static ConstraintSet generateRandomEllipsoidBasedConstraintSet(boolean includeEqualityConstraints, boolean useEqualityConstraintsAsInequalityConstraints)
   {
      int dimensionality = 2 + random.nextInt(30);
      int inequalityConstraints = 1 + random.nextInt(30);
      int equalityConstraints = includeEqualityConstraints ? 1 + random.nextInt(dimensionality - 1) : 0;

      double radiusSquared = 1.0 + 100.0 * random.nextDouble();
      double[] alphas = new double[dimensionality];
      for (int j = 0; j < alphas.length; j++)
      {
         alphas[j] = 1.0 + 30.0 * random.nextDouble();
      }

      ConstraintSet constraintSet = new ConstraintSet();
      addInequalityConstraints(radiusSquared, alphas, constraintSet, inequalityConstraints, dimensionality);

      if (includeEqualityConstraints)
      {
         if (useEqualityConstraintsAsInequalityConstraints)
         {
            DMatrixRMaj Aeq = new DMatrixRMaj(0);
            DMatrixRMaj beq = new DMatrixRMaj(0);
            addEqualityConstraints(radiusSquared, alphas, Aeq, beq, equalityConstraints, dimensionality);

            int constraints = inequalityConstraints + 2 * equalityConstraints;
            DMatrixRMaj A = constraintSet.inequalityMatrix;
            DMatrixRMaj b = constraintSet.inequalityVector;
            A.reshape(constraints, dimensionality, true);
            b.reshape(constraints, 1, true);

            MatrixTools.setMatrixBlock(A, inequalityConstraints,                    0, Aeq, 0, 0, Aeq.getNumRows(), Aeq.getNumCols(), 1.0);
            MatrixTools.setMatrixBlock(A, inequalityConstraints + Aeq.getNumRows(), 0, Aeq, 0, 0, Aeq.getNumRows(), Aeq.getNumCols(), -1.0);

            MatrixTools.setMatrixBlock(b, inequalityConstraints,                    0, beq, 0, 0, beq.getNumRows(), beq.getNumCols(), 1.0);
            MatrixTools.setMatrixBlock(b, inequalityConstraints + beq.getNumRows(), 0, beq, 0, 0, beq.getNumRows(), beq.getNumCols(), -1.0);
         }
         else
         {
            addEqualityConstraints(radiusSquared, alphas, constraintSet.equalityMatrix, constraintSet.equalityVector, equalityConstraints, dimensionality);
         }
      }

      return constraintSet;
   }

   private static void addInequalityConstraints(double radiusSquared, double[] alphas, ConstraintSet constraintSet, int numberOfInequalityConstraints, int dimensionality)
   {
      constraintSet.inequalityMatrix.reshape(numberOfInequalityConstraints, dimensionality);
      constraintSet.inequalityVector.reshape(numberOfInequalityConstraints, 1);

      for (int i = 0; i < numberOfInequalityConstraints; i++)
      {
         // compute initial point on curve
         double[] initialPoint = generatePointOnEllipsoid(dimensionality, radiusSquared, alphas);

         // compute gradient at this point
         double[] gradient = new double[dimensionality];
         for (int j = 0; j < dimensionality; j++)
         {
            gradient[j] = alphas[j] * initialPoint[j];
         }

         double bValue = 0.0;
         for (int k = 0; k < dimensionality; k++)
         {
            bValue += gradient[k] * initialPoint[k];
         }

         for (int j = 0; j < dimensionality; j++)
         {
            constraintSet.inequalityMatrix.set(i, j, gradient[j]);
         }

         constraintSet.inequalityVector.set(i, 0, bValue);
      }
   }

   private static void addEqualityConstraints(double radiusSquared, double[] alphas, DMatrixRMaj Aeq, DMatrixRMaj beq, int numberOfEqualityConstraints, int dimensionality)
   {
      Aeq.reshape(numberOfEqualityConstraints, dimensionality);
      beq.reshape(numberOfEqualityConstraints, 1);

      double[] interiorPoint = generatePointOnEllipsoid(dimensionality, radiusSquared, alphas);
      double scale = 0.9 * random.nextDouble();
      for (int i = 0; i < interiorPoint.length; i++)
      {
         interiorPoint[i] = scale * interiorPoint[i];
      }

      for (int i = 0; i < numberOfEqualityConstraints; i++)
      {
         double normDotPoint = 0.0;
         double[] normal = generateRandomVectorForPlaneNormal(dimensionality);

         for (int j = 0; j < dimensionality; j++)
         {
            normDotPoint += normal[j] * interiorPoint[j];
         }

         beq.set(i, 0, normDotPoint);

         for (int j = 0; j < dimensionality; j++)
         {
            Aeq.set(i, j, normal[j]);
         }
      }
   }

   private static double[] generatePointOnEllipsoid(int dimensionality, double radiusSquared, double[] alphas)
   {
      double[] initialPoint = new double[dimensionality];
      double remainingPosValue = radiusSquared;
      for (int k = 0; k < dimensionality - 1; k++)
      {
         double alphaXSq = EuclidCoreRandomTools.nextDouble(random, 0.0, remainingPosValue * 0.99 / alphas[k]);
         remainingPosValue -= alphaXSq;
         initialPoint[k] = Math.sqrt(alphaXSq / alphas[k]);
      }

      initialPoint[dimensionality - 1] = Math.sqrt(remainingPosValue / alphas[dimensionality - 1]);
      return initialPoint;
   }

   private static double[] generateRandomVectorForPlaneNormal(int dimensionality)
   {
      double[] v = new double[dimensionality];
      double sumSq = 0.0;

      for (int i = 0; i < dimensionality; i++)
      {
         v[i] = EuclidCoreRandomTools.nextDouble(random, 1.0);
         sumSq += MathTools.square(v[i]);
      }

      double norm = Math.sqrt(sumSq);
      if (norm < 1e-3)
      {
         v[0] = 1.0;
      }
      else
      {
         for (int i = 0; i < dimensionality; i++)
         {
            v[i] /= Math.sqrt(1.0 / norm);
         }
      }

      return v;
   }

   /**
    * Sets A and b matrices to random constraint set
    */
   private static ConstraintSet generateRandomConstraints()
   {
      int dimensionality = 2 + random.nextInt(40);
      int constraints = 1 + random.nextInt(40);

      double minMaxConstraint = 10.0;
      ConstraintSet constraintSet = new ConstraintSet();
      constraintSet.inequalityMatrix.reshape(constraints, dimensionality);
      constraintSet.inequalityVector.reshape(constraints, 1);

      for (int i = 0; i < constraints; i++)
      {
         constraintSet.inequalityVector.set(i, EuclidCoreRandomTools.nextDouble(random, minMaxConstraint));

         for (int j = 0; j < dimensionality; j++)
         {
            constraintSet.inequalityMatrix.set(i, j, EuclidCoreRandomTools.nextDouble(random, minMaxConstraint));
         }
      }

      return constraintSet;
   }

   private static DMatrixRMaj generateRandomCostVector(int dimensionality)
   {
      DMatrixRMaj c = new DMatrixRMaj(dimensionality, 1);
      double minMaxEntry = 10.0;
      double l1Norm = 0.0;

      for (int i = 0; i < dimensionality; i++)
      {
         double entry = EuclidCoreRandomTools.nextDouble(random, minMaxEntry);
         l1Norm += Math.abs(entry);

         c.set(i, 0, entry);
      }

      double minNorm = 1e-3;
      if (l1Norm < minNorm)
      {
         c.set(0, 0, 1.0);
      }

      return c;
   }

   private static double[] solveWithApacheCommons(ConstraintSet constraintSet, DMatrixRMaj c, Relationship constraintRelationship)
   {
      SimplexSolver apacheSolver = new SimplexSolver();

      double[] directionToMaximize = Arrays.copyOf(c.getData(), c.getNumRows());
      LinearObjectiveFunction objectiveFunction = new LinearObjectiveFunction(directionToMaximize, 0.0);

      DMatrixRMaj inequalityMatrix = constraintSet.inequalityMatrix;
      DMatrixRMaj inequalityVector = constraintSet.inequalityVector;
      DMatrixRMaj equalityMatrix = constraintSet.equalityMatrix;
      DMatrixRMaj equalityVector = constraintSet.equalityVector;

      List<LinearConstraint> constraintList = new ArrayList<>();
      for (int i = 0; i < inequalityMatrix.getNumRows(); i++)
      {
         double[] constraint = new double[inequalityMatrix.getNumCols()];
         for (int j = 0; j < inequalityMatrix.getNumCols(); j++)
         {
            constraint[j] = inequalityMatrix.get(i, j);
         }

         constraintList.add(new LinearConstraint(constraint, constraintRelationship, inequalityVector.get(i)));
      }

      for (int i = 0; i < inequalityMatrix.getNumCols(); i++)
      {
         double[] nonNegativeConstraint = new double[inequalityMatrix.getNumCols()];
         nonNegativeConstraint[i] = 1.0;
         constraintList.add(new LinearConstraint(nonNegativeConstraint, Relationship.GEQ, 0.0));
      }

      for (int i = 0; i < equalityMatrix.getNumRows(); i++)
      {
         double[] constraint = new double[equalityMatrix.getNumCols()];
         for (int j = 0; j < equalityMatrix.getNumCols(); j++)
         {
            constraint[j] = equalityMatrix.get(i, j);
         }

         constraintList.add(new LinearConstraint(constraint, Relationship.EQ, equalityVector.get(i)));
      }

      try
      {
         return apacheSolver.optimize(new MaxIter(1000), objectiveFunction, new LinearConstraintSet(constraintList), GoalType.MAXIMIZE).getPoint();
      }
      catch (Exception e)
      {
         return null;
      }
   }
}
