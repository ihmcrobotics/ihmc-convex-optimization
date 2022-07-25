package us.ihmc.convexOptimization.linearProgram;

import org.apache.commons.lang3.tuple.Pair;
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

   @Test
   public void testEllipsoidBasedMaxConstraint()
   {
      int tests = 400;
      int costVectorsPerProblem = 10;
      LinearProgramSolver customSolver = new LinearProgramSolver();

      for (int i = 0; i < tests; i++)
      {
         Pair<DMatrixRMaj, DMatrixRMaj> constraintSet = generateRandomEllipsoidBasedConstraintSet(false);
         DMatrixRMaj A = constraintSet.getLeft();
         DMatrixRMaj b = constraintSet.getRight();

         for (int j = 0; j < costVectorsPerProblem; j++)
         {
            DMatrixRMaj costVector = generateRandomCostVector(A.getNumCols());

            // SOLVE WITH APACHE //
            double[] apacheCommonsSolution = solveWithApacheCommons(A, b, costVector, Relationship.LEQ);

            // SOLVER WITH CUSTOM IMPL USING SIMPLEX //
            DMatrixRMaj simplexSolution = new DMatrixRMaj(0);
            boolean foundSimplexSolution = customSolver.solve(costVector, A, b, simplexSolution, SolverMethod.SIMPLEX);

            // SOLVER WITH CUSTOM IMPL USING CRISS CROSS //
            DMatrixRMaj crissCrossSolution = new DMatrixRMaj(0);
            boolean foundCrissCrossSolution = customSolver.solve(costVector, A, b, crissCrossSolution, SolverMethod.CRISS_CROSS);

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
   public void testEllipsoidBasedMaxAndEqualityConstraints()
   {
      int tests = 100;
      int costVectorsPerProblem = 10;
      LinearProgramSolver customSolver = new LinearProgramSolver();

      for (int i = 0; i < tests; i++)
      {
         Pair<DMatrixRMaj, DMatrixRMaj> constraintPlanes = generateRandomEllipsoidBasedConstraintSet(true);
         DMatrixRMaj A = constraintPlanes.getLeft();
         DMatrixRMaj b = constraintPlanes.getRight();

         for (int j = 0; j < costVectorsPerProblem; j++)
         {
            DMatrixRMaj costVector = generateRandomCostVector(A.getNumCols());

            // SOLVE WITH APACHE //
            double[] apacheCommonsSolution = solveWithApacheCommons(A, b, costVector, Relationship.LEQ);

            // SOLVER WITH CUSTOM IMPL USING SIMPLEX //
            DMatrixRMaj simplexSolution = new DMatrixRMaj(0);
            boolean foundSimplexSolution = customSolver.solve(costVector, A, b, simplexSolution, SolverMethod.SIMPLEX);

            // SOLVER WITH CUSTOM IMPL USING CRISS CROSS //
            DMatrixRMaj crissCrossSolution = new DMatrixRMaj(0);
            boolean foundCrissCrossSolution = customSolver.solve(costVector, A, b, crissCrossSolution, SolverMethod.CRISS_CROSS);

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
         Pair<DMatrixRMaj, DMatrixRMaj> constraintPlanes = generateRandomEllipsoidBasedConstraintSet(false);
         DMatrixRMaj A = constraintPlanes.getLeft();
         DMatrixRMaj b = constraintPlanes.getRight();
         CommonOps_DDRM.scale(-1.0, A);
         CommonOps_DDRM.scale(-1.0, b);

         for (int j = 0; j < costVectorsPerProblem; j++)
         {
            DMatrixRMaj costVector = generateRandomCostVector(A.getNumCols());

            // SOLVE WITH APACHE //
            double[] apacheCommonsSolution = solveWithApacheCommons(A, b, costVector, Relationship.LEQ);

            // SOLVER WITH CUSTOM IMPL USING SIMPLEX //
            DMatrixRMaj simplexSolution = new DMatrixRMaj(0);
            boolean foundSimplexSolution = customSolver.solve(costVector, A, b, simplexSolution, SolverMethod.SIMPLEX);

            // SOLVER WITH CUSTOM IMPL USING CRISS CROSS //
            DMatrixRMaj crissCrossSolution = new DMatrixRMaj(0);
            boolean foundCrissCrossSolution = customSolver.solve(costVector, A, b, crissCrossSolution, SolverMethod.CRISS_CROSS);

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
   public void testLPWithEqualityConstraint()
   {
      LinearProgramSolver customSolver = new LinearProgramSolver();
      Pair<Pair<DMatrixRMaj, DMatrixRMaj>, Pair<DMatrixRMaj, DMatrixRMaj>> constraintSet = generateRandomEllipsoidBasedConstraintSetSeparated();

      DMatrixRMaj A = constraintSet.getLeft().getLeft();
      DMatrixRMaj b = constraintSet.getLeft().getRight();
      DMatrixRMaj C = constraintSet.getRight().getLeft();
      DMatrixRMaj d = constraintSet.getRight().getRight();

      DMatrixRMaj costVector = generateRandomCostVector(A.getNumCols());
      DMatrixRMaj solutionToPack = new DMatrixRMaj(0);

      //SOLVE USING EQUALITY PASS IN CONSTRUCTOR SIMPLEX SOLUTION //
      boolean foundSolution = customSolver.solve(costVector, A, b, C, d, solutionToPack);

      //SOLVE WITH APACHE //
      // ** this is where I need help **
      // How to pass A, b, C, d into apache solution without pre-packing as A' b'
      double[] apacheCommonsSolution = solveWithApacheCommons(A, b, costVector, Relationship.LEQ/* Doesn't accept equality contraint matrix */);

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
    * Sets inequality matrices to be planes that are tangent to an ellipsoid. Sets equality matrices to converge at a point interior to the ellipsoid.
    */
   private static Pair<DMatrixRMaj, DMatrixRMaj> generateRandomEllipsoidBasedConstraintSet(boolean includeEqualityConstraints)
   {
      int dimensionality = 2 + random.nextInt(30);
      int inequalityConstraints = 1 + random.nextInt(30);

      double radiusSquared = 1.0 + 100.0 * random.nextDouble();
      double[] alphas = new double[dimensionality];
      for (int j = 0; j < alphas.length; j++)
      {
         alphas[j] = 1.0 + 30.0 * random.nextDouble();
      }

      DMatrixRMaj Ain = new DMatrixRMaj(inequalityConstraints, dimensionality);
      DMatrixRMaj bin = new DMatrixRMaj(inequalityConstraints, 1);

      for (int i = 0; i < inequalityConstraints; i++)
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
            Ain.set(i, j, gradient[j]);
         }

         bin.set(i, 0, bValue);
      }

      if (!includeEqualityConstraints)
      {
         return Pair.of(Ain, bin);
      }

      int equalityConstraints = 1 + random.nextInt(dimensionality - 1);
      DMatrixRMaj Aeq = new DMatrixRMaj(equalityConstraints, dimensionality);
      DMatrixRMaj beq = new DMatrixRMaj(equalityConstraints, 1);

      double[] interiorPoint = generatePointOnEllipsoid(dimensionality, radiusSquared, alphas);
      double scale = 0.9 * random.nextDouble();
      for (int i = 0; i < interiorPoint.length; i++)
      {
         interiorPoint[i] = scale * interiorPoint[i];
      }

      for (int i = 0; i < equalityConstraints; i++)
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

      int constraints = inequalityConstraints + 2 * equalityConstraints;
      DMatrixRMaj A = new DMatrixRMaj(constraints, dimensionality);
      DMatrixRMaj b = new DMatrixRMaj(constraints, 1);

      MatrixTools.setMatrixBlock(A, 0, 0, Ain, 0, 0, Ain.getNumRows(), Ain.getNumCols(), 1.0);
      MatrixTools.setMatrixBlock(A, Ain.getNumRows(), 0, Aeq, 0, 0, Aeq.getNumRows(), Aeq.getNumCols(), 1.0);
      MatrixTools.setMatrixBlock(A, Ain.getNumRows() + Aeq.getNumRows(), 0, Aeq, 0, 0, Aeq.getNumRows(), Aeq.getNumCols(), -1.0);

      MatrixTools.setMatrixBlock(b, 0, 0, bin, 0, 0, bin.getNumRows(), bin.getNumCols(), 1.0);
      MatrixTools.setMatrixBlock(b, bin.getNumRows(), 0, beq, 0, 0, beq.getNumRows(), beq.getNumCols(), 1.0);
      MatrixTools.setMatrixBlock(b, bin.getNumRows() + beq.getNumRows(), 0, beq, 0, 0, beq.getNumRows(), beq.getNumCols(), -1.0);

      return Pair.of(A, b);
   }

   /**
    * Generates inequality constraints and equality constraints,
    * returns as separated matrices Ax <= b & Cx = d
    */
   private static Pair< Pair<DMatrixRMaj, DMatrixRMaj> , Pair<DMatrixRMaj, DMatrixRMaj> > generateRandomEllipsoidBasedConstraintSetSeparated()
   {
      int dimensionality = 2 + random.nextInt(30);
      int inequalityConstraints = 1 + random.nextInt(30);

      double radiusSquared = 1.0 + 100.0 * random.nextDouble();
      double[] alphas = new double[dimensionality];
      for (int j = 0; j < alphas.length; j++)
      {
         alphas[j] = 1.0 + 30.0 * random.nextDouble();
      }

      DMatrixRMaj Ain = new DMatrixRMaj(inequalityConstraints, dimensionality);
      DMatrixRMaj bin = new DMatrixRMaj(inequalityConstraints, 1);

      for (int i = 0; i < inequalityConstraints; i++)
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
            Ain.set(i, j, gradient[j]);
         }

         bin.set(i, 0, bValue);
      }


      int equalityConstraints = 1 + random.nextInt(dimensionality - 1);
      DMatrixRMaj Aeq = new DMatrixRMaj(equalityConstraints, dimensionality);
      DMatrixRMaj beq = new DMatrixRMaj(equalityConstraints, 1);

      double[] interiorPoint = generatePointOnEllipsoid(dimensionality, radiusSquared, alphas);
      double scale = 0.9 * random.nextDouble();
      for (int i = 0; i < interiorPoint.length; i++)
      {
         interiorPoint[i] = scale * interiorPoint[i];
      }

      for (int i = 0; i < equalityConstraints; i++)
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

      return Pair.of(Pair.of(Ain, bin), Pair.of(Aeq, beq));
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
