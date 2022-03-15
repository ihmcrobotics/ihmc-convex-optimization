package us.ihmc.convexOptimization.linearProgram;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.math3.optim.MaxIter;
import org.apache.commons.math3.optim.linear.*;
import org.apache.commons.math3.optim.nonlinear.scalar.GoalType;
import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import us.ihmc.euclid.tools.EuclidCoreRandomTools;
import us.ihmc.euclid.tools.EuclidCoreTools;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class LinearProgramSolverTest
{
   private static final Random random = new Random(349034);
   private static final double epsilon = 1e-5;

   @Test
   public void testEllipsoidBasedMaxConstraint()
   {
      int tests = 400;
      int costVectorsPerProblem = 10;
      LinearProgramSolver customSolver = new LinearProgramSolver();

      for (int i = 0; i < tests; i++)
      {
         Pair<DMatrixRMaj, DMatrixRMaj> constraintPlanes = generatePlanesTangentToRandomEllipsoid();

         for (int j = 0; j < costVectorsPerProblem; j++)
         {
            DMatrixRMaj costVector = generateRandomCostVector(constraintPlanes.getLeft().getNumCols());

            // SOLVE WITH APACHE //
            double[] apacheCommonsSolution = solveWithApacheCommons(constraintPlanes.getLeft(), constraintPlanes.getRight(), costVector, Relationship.LEQ);

            // SOLVER WITH CUSTOM IMPL USING SIMPLEX //
            DMatrixRMaj simplexSolution = new DMatrixRMaj(0);
            boolean foundSimplexSolution = customSolver.solve(costVector, constraintPlanes.getLeft(), constraintPlanes.getRight(), simplexSolution, SolverMethod.SIMPLEX);

            // SOLVER WITH CUSTOM IMPL USING CRISS CROSS //
            DMatrixRMaj crissCrossSolution = new DMatrixRMaj(0);
            boolean foundCrissCrossSolution = customSolver.solve(costVector, constraintPlanes.getLeft(), constraintPlanes.getRight(), crissCrossSolution, SolverMethod.CRISS_CROSS);

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
   }

   @Test
   public void testEllipsoidBasedMinConstraint()
   {
      int tests = 400;
      int costVectorsPerProblem = 10;
      LinearProgramSolver customSolver = new LinearProgramSolver();

      for (int i = 0; i < tests; i++)
      {
         Pair<DMatrixRMaj, DMatrixRMaj> constraintPlanes = generatePlanesTangentToRandomEllipsoid();
         CommonOps_DDRM.scale(-1.0, constraintPlanes.getLeft());
         CommonOps_DDRM.scale(-1.0, constraintPlanes.getRight());

         for (int j = 0; j < costVectorsPerProblem; j++)
         {
            DMatrixRMaj costVector = generateRandomCostVector(constraintPlanes.getLeft().getNumCols());

            // SOLVE WITH APACHE //
            double[] apacheCommonsSolution = solveWithApacheCommons(constraintPlanes.getLeft(), constraintPlanes.getRight(), costVector, Relationship.LEQ);

            // SOLVER WITH CUSTOM IMPL USING SIMPLEX //
            DMatrixRMaj simplexSolution = new DMatrixRMaj(0);
            boolean foundSimplexSolution = customSolver.solve(costVector, constraintPlanes.getLeft(), constraintPlanes.getRight(), simplexSolution, SolverMethod.SIMPLEX);

            // SOLVER WITH CUSTOM IMPL USING CRISS CROSS //
            DMatrixRMaj crissCrossSolution = new DMatrixRMaj(0);
            boolean foundCrissCrossSolution = customSolver.solve(costVector, constraintPlanes.getLeft(), constraintPlanes.getRight(), crissCrossSolution, SolverMethod.CRISS_CROSS);

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
   }

   @Test
   public void testRandomLPs()
   {
      int tests = 200;
      int costVectorsPerProblem = 10;
      LinearProgramSolver customSolver = new LinearProgramSolver();

      for (int i = 0; i < tests; i++)
      {
         Pair<DMatrixRMaj, DMatrixRMaj> constraintPlanes = generateRandomConstraints();

         for (int j = 0; j < costVectorsPerProblem; j++)
         {
            DMatrixRMaj costVector = generateRandomCostVector(constraintPlanes.getLeft().getNumCols());

            // SOLVE WITH APACHE //
            double[] apacheCommonsSolution = solveWithApacheCommons(constraintPlanes.getLeft(), constraintPlanes.getRight(), costVector, Relationship.LEQ);

            // SOLVER WITH CUSTOM IMPL USING SIMPLEX //
            DMatrixRMaj simplexSolution = new DMatrixRMaj(0);
            boolean foundSimplexSolution = customSolver.solve(costVector, constraintPlanes.getLeft(), constraintPlanes.getRight(), simplexSolution, SolverMethod.SIMPLEX);

            // SOLVER WITH CUSTOM IMPL USING CRISS CROSS //
            DMatrixRMaj crissCrossSolution = new DMatrixRMaj(0);
            boolean foundCrissCrossSolution = customSolver.solve(costVector, constraintPlanes.getLeft(), constraintPlanes.getRight(), crissCrossSolution, SolverMethod.CRISS_CROSS);

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
   }

   /**
    * Sets A and b matrices to be planes that are tangent to an ellipsoid
    */
   private static Pair<DMatrixRMaj, DMatrixRMaj> generatePlanesTangentToRandomEllipsoid()
   {
      int dimensionality = 2 + random.nextInt(30);
      int constraints = 1 + random.nextInt(30);

      double radiusSquared = 1.0 + 100.0 * random.nextDouble();
      double[] alphas = new double[dimensionality];
      for (int j = 0; j < alphas.length; j++)
      {
         alphas[j] = 1.0 + 30.0 * random.nextDouble();
      }

      DMatrixRMaj A = new DMatrixRMaj(constraints, dimensionality);
      DMatrixRMaj b = new DMatrixRMaj(constraints, 1);

      for (int i = 0; i < constraints; i++)
      {
         // compute initial point on curve
         double[] initialPoint = new double[dimensionality];
         double remainingPosValue = radiusSquared;
         for (int k = 0; k < dimensionality - 1; k++)
         {
            double alphaXSq = EuclidCoreRandomTools.nextDouble(random, 0.0, remainingPosValue * 0.99 / alphas[k]);
            remainingPosValue -= alphaXSq;
            initialPoint[k] = Math.sqrt(alphaXSq / alphas[k]);
         }

         initialPoint[dimensionality - 1] = Math.sqrt(remainingPosValue / alphas[dimensionality - 1]);

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
            A.set(i, j, gradient[j]);
         }

         b.set(i, 0, bValue);
      }

      return Pair.of(A, b);
   }

   /**
    * Sets A and b matrices to be planes that are tangent to an ellipsoid
    */
   private static Pair<DMatrixRMaj, DMatrixRMaj> generateRandomConstraints()
   {
      int dimensionality = 2 + random.nextInt(40);
      int constraints = 1 + random.nextInt(40);

      double minMaxConstraint = 10.0;
      DMatrixRMaj A = new DMatrixRMaj(constraints, dimensionality);
      DMatrixRMaj b = new DMatrixRMaj(constraints, 1);

      for (int i = 0; i < constraints; i++)
      {
         b.set(i, EuclidCoreRandomTools.nextDouble(random, minMaxConstraint));

         for (int j = 0; j < dimensionality; j++)
         {
            A.set(i, j, EuclidCoreRandomTools.nextDouble(random, minMaxConstraint));
         }
      }

      return Pair.of(A, b);

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

   private static double[] solveWithApacheCommons(DMatrixRMaj A, DMatrixRMaj b, DMatrixRMaj c, Relationship constraintRelationship)
   {
      SimplexSolver apacheSolver = new SimplexSolver();

      double[] directionToMaximize = Arrays.copyOf(c.getData(), c.getNumRows());
      LinearObjectiveFunction objectiveFunction = new LinearObjectiveFunction(directionToMaximize, 0.0);

      List<LinearConstraint> constraintList = new ArrayList<>();
      for (int i = 0; i < A.getNumRows(); i++)
      {
         double[] constraint = new double[A.getNumCols()];
         for (int j = 0; j < A.getNumCols(); j++)
         {
            constraint[j] = A.get(i, j);
         }

         constraintList.add(new LinearConstraint(constraint, constraintRelationship, b.get(i)));
      }

      for (int i = 0; i < A.getNumCols(); i++)
      {
         double[] nonNegativeConstraint = new double[A.getNumCols()];
         nonNegativeConstraint[i] = 1.0;
         constraintList.add(new LinearConstraint(nonNegativeConstraint, Relationship.GEQ, 0.0));
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
