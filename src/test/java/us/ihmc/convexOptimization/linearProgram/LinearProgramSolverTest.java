package us.ihmc.convexOptimization.linearProgram;

import gnu.trove.list.array.TIntArrayList;
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
   public void testSolutionDictionaryIndices()
   {
      LinearProgramSolver solver = new LinearProgramSolver();

      for (SolverMethod solverMethod : SolverMethod.values())
      {
         TIntArrayList nonBasisIndices = solver.getNonBasisIndices();
         TIntArrayList basisIndices = solver.getBasisIndices();

         DMatrixRMaj cost = new DMatrixRMaj(2, 1);
         DMatrixRMaj A = new DMatrixRMaj(3, 2);
         DMatrixRMaj b = new DMatrixRMaj(3, 1);

         A.set(0, 0, 1.0);
         A.set(1, 1, 1.0);
         A.set(2, 0, 1.0);
         A.set(2, 1, 1.0);

         b.set(0, 0, 2.0);
         b.set(1, 0, 2.0);
         b.set(2, 0, 3.0);

         int nonNegConstraint1 = 1;
         int nonNegConstraint2 = 2;
         int AConstraint1 = 3;
         int AConstraint2 = 4;
         int AConstraint3 = 5;

         DMatrixRMaj solution = new DMatrixRMaj(2, 1);

         /* Pointing to the origin */
         cost.set(0, 0, -1.0);
         cost.set(1, 0, -1.0);
         solver.solve(cost, A, b, solution, solverMethod);
         Assertions.assertEquals(basisIndices.size(), 1 + A.getNumRows());
         Assertions.assertEquals(nonBasisIndices.size(), 1 + A.getNumCols());
         Assertions.assertTrue(nonBasisIndices.contains(nonNegConstraint1));
         Assertions.assertTrue(nonBasisIndices.contains(nonNegConstraint2));
         Assertions.assertTrue(basisIndices.contains(AConstraint1));
         Assertions.assertTrue(basisIndices.contains(AConstraint2));
         Assertions.assertTrue(basisIndices.contains(AConstraint3));

         /* Pointing right and down */
         cost.set(0, 0, 1.0);
         cost.set(1, 0, -1.0);
         solver.solve(cost, A, b, solution, solverMethod);
         Assertions.assertTrue(nonBasisIndices.contains(nonNegConstraint2));
         Assertions.assertTrue(nonBasisIndices.contains(AConstraint1));
         Assertions.assertTrue(basisIndices.contains(nonNegConstraint1));
         Assertions.assertTrue(basisIndices.contains(AConstraint2));
         Assertions.assertTrue(basisIndices.contains(AConstraint3));

         /* Pointing right and a little up */
         cost.set(0, 0, 1.0);
         cost.set(1, 0, 0.01);
         solver.solve(cost, A, b, solution, solverMethod);
         Assertions.assertTrue(nonBasisIndices.contains(AConstraint1));
         Assertions.assertTrue(nonBasisIndices.contains(AConstraint3));
         Assertions.assertTrue(basisIndices.contains(nonNegConstraint1));
         Assertions.assertTrue(basisIndices.contains(nonNegConstraint2));
         Assertions.assertTrue(basisIndices.contains(AConstraint2));

         /* Pointing up and a little right */
         cost.set(0, 0, 0.01);
         cost.set(1, 0, 1.0);
         solver.solve(cost, A, b, solution, solverMethod);
         Assertions.assertTrue(nonBasisIndices.contains(AConstraint2));
         Assertions.assertTrue(nonBasisIndices.contains(AConstraint3));
         Assertions.assertTrue(basisIndices.contains(nonNegConstraint1));
         Assertions.assertTrue(basisIndices.contains(nonNegConstraint2));
         Assertions.assertTrue(basisIndices.contains(AConstraint1));

         /* Pointing up and left */
         cost.set(0, 0, -1.0);
         cost.set(1, 0, 1.0);
         solver.solve(cost, A, b, solution, solverMethod);
         Assertions.assertTrue(nonBasisIndices.contains(nonNegConstraint1));
         Assertions.assertTrue(nonBasisIndices.contains(AConstraint2));
         Assertions.assertTrue(basisIndices.contains(nonNegConstraint2));
         Assertions.assertTrue(basisIndices.contains(AConstraint1));
         Assertions.assertTrue(basisIndices.contains(AConstraint3));
      }
   }

   @Test
   public void testNonBasisIndicesSaturateConstraints()
   {
      int numTests = 100;
      LinearProgramSolver lpSolver = new LinearProgramSolver();

      for (int i = 0; i < numTests; i++)
      {
         ConstraintSet constraintSet = generateRandomEllipsoidBasedConstraintSet(false, false);
         DMatrixRMaj costVector = generateRandomCostVector(constraintSet.inequalityMatrix.getNumCols());
         DMatrixRMaj solution = new DMatrixRMaj(0);

         boolean foundSolution = lpSolver.solve(costVector, constraintSet.inequalityMatrix, constraintSet.inequalityVector, solution, SolverMethod.SIMPLEX);
         TIntArrayList nonBasisIndices = lpSolver.getNonBasisIndices();

         if (foundSolution)
         {
            TIntArrayList saturatedConstraintIndices = new TIntArrayList();

            for (int j = 1; j < nonBasisIndices.size(); j++)
            {
               int nonBasisIndex = nonBasisIndices.get(j);
               if (lpSolver.isNonNegativeConstraint(nonBasisIndex))
               { // Non-negative constraint is saturated, check that solution value is 0.0
                  int saturatedVariableIndex = LinearProgramSolver.toVariableIndex(nonBasisIndex);
                  Assertions.assertTrue(EuclidCoreTools.epsilonEquals(solution.get(saturatedVariableIndex), 0.0, epsilon));
               }
               else
               { // Canonical-form matrix constraint, add to list
                  int saturatedConstraintIndex = lpSolver.toConstraintIndex(nonBasisIndex);
                  saturatedConstraintIndices.add(saturatedConstraintIndex);
               }
            }

            int numSaturatedMatrixConstraints = saturatedConstraintIndices.size();
            DMatrixRMaj A_saturated = new DMatrixRMaj(numSaturatedMatrixConstraints, constraintSet.inequalityMatrix.getNumCols());
            DMatrixRMaj b_saturated = new DMatrixRMaj(numSaturatedMatrixConstraints, 1);

            for (int j = 0; j < numSaturatedMatrixConstraints; j++)
            {
               int saturatedConstraintIndex = saturatedConstraintIndices.get(j);
               MatrixTools.setMatrixBlock(A_saturated, j, 0, constraintSet.inequalityMatrix, saturatedConstraintIndex, 0, 1, constraintSet.inequalityMatrix.getNumCols(), 1.0);
               b_saturated.set(j, 0, constraintSet.inequalityVector.get(saturatedConstraintIndex, 0));
            }

            DMatrixRMaj b_solution = new DMatrixRMaj(constraintSet.inequalityMatrix.getNumCols(), 1);
            CommonOps_DDRM.mult(A_saturated, solution, b_solution);

            for (int j = 0; j < numSaturatedMatrixConstraints; j++)
            {
               Assertions.assertTrue(EuclidCoreTools.epsilonEquals(b_saturated.get(j, 0), b_solution.get(j, 0), epsilon));
            }
         }
      }
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

   @Test
   public void testSolveForFixedBasis()
   {
      int tests = 200;

      // debug to check that this is testing something
//      int numSameBasis = 0;

      for (int i = 0; i < tests; i++)
      {
         ConstraintSet constraintSet = generateRandomEllipsoidBasedConstraintSet(false, false);

         DMatrixRMaj costVector = generateRandomCostVector(constraintSet.inequalityMatrix.getNumCols());
         DMatrixRMaj expectedSolution = new DMatrixRMaj(0);

         LinearProgramSolver solver = new LinearProgramSolver();
         solver.solve(costVector, constraintSet.inequalityMatrix, constraintSet.inequalityVector, expectedSolution);
         TIntArrayList originalBasisIndices = new TIntArrayList(solver.getBasisIndices());

         DMatrixRMaj mutatedInequalityMatrix = new DMatrixRMaj(constraintSet.inequalityMatrix);
         mutateInequalityMatrix(constraintSet.inequalityMatrix, mutatedInequalityMatrix);
         solver.solve(costVector, mutatedInequalityMatrix, constraintSet.inequalityVector, expectedSolution);
         TIntArrayList mutatedBasisIndices = new TIntArrayList(solver.getBasisIndices());

         if (!containsSameElements(originalBasisIndices, mutatedBasisIndices))
            continue;

//         numSameBasis++;

         DMatrixRMaj calculatedSolution = new DMatrixRMaj(0);
         solver.solveForFixedBasis(mutatedInequalityMatrix, constraintSet.inequalityVector, originalBasisIndices, calculatedSolution);

         for (int j = 0; j < expectedSolution.getNumRows(); j++)
         {
            double epsilon = 1e-12;
            Assertions.assertTrue(EuclidCoreTools.epsilonEquals(calculatedSolution.get(j), expectedSolution.get(j), epsilon));
         }
      }

//      System.out.println("numSame: " + numSameBasis + "/" + tests);
   }

   @Test
   public void testComputeSensitivity()
   {
      int tests = 200;
      double theta = 1e-8;

      // debug to check that this is testing something
      //      int numSameBasis = 0;

      for (int i = 0; i < tests; i++)
      {
         ConstraintSet constraintSet = generateRandomEllipsoidBasedConstraintSet(false, false);
         DMatrixRMaj costVector = generateRandomCostVector(constraintSet.inequalityMatrix.getNumCols());
         DMatrixRMaj solution = new DMatrixRMaj(0);

         LinearProgramSolver solver = new LinearProgramSolver();
         solver.solve(costVector, constraintSet.inequalityMatrix, constraintSet.inequalityVector, solution);

         DMatrixRMaj z0 = new DMatrixRMaj(0);
         CommonOps_DDRM.multTransA(costVector, solution, z0);
         TIntArrayList originalBasisIndices = new TIntArrayList(solver.getBasisIndices());

         DMatrixRMaj constraintMatrixVariation = generateRandomConstraintMatrixVariation(constraintSet.inequalityMatrix.getNumRows(),
                                                                                         constraintSet.inequalityMatrix.getNumCols());
         double expectedSensitivity = solver.computeSensitivity(constraintMatrixVariation);

         CommonOps_DDRM.scale(theta, constraintMatrixVariation);
         DMatrixRMaj modifiedConstraintMatrix = new DMatrixRMaj(constraintSet.inequalityMatrix);
         CommonOps_DDRM.addEquals(modifiedConstraintMatrix, constraintMatrixVariation);

         solver.solve(costVector, modifiedConstraintMatrix, constraintSet.inequalityVector, solution);
         TIntArrayList mutatedBasisIndices = new TIntArrayList(solver.getBasisIndices());

         if (!containsSameElements(originalBasisIndices, mutatedBasisIndices))
            continue;

         //         numSameBasis++;

         DMatrixRMaj directSensitivitySolution = new DMatrixRMaj(0);
         solver.solve(costVector, modifiedConstraintMatrix, constraintSet.inequalityVector, directSensitivitySolution);

         DMatrixRMaj zi = new DMatrixRMaj(0);
         CommonOps_DDRM.multTransA(costVector, directSensitivitySolution, zi);

         double computedSensitivity = (zi.get(0) - z0.get(0)) / theta;
         Assertions.assertTrue(Math.abs(expectedSensitivity - computedSensitivity) < 1.0e-3, "Expected and computed sensitivity do not match.");

//         System.out.println((expectedSensitivity - computedSensitivity) + "\n " + expectedSensitivity + "\n " + computedSensitivity);
//         System.out.println();
      }

      //      System.out.println("numSame: " + numSameBasis + "/" + tests);
   }

   private static void mutateInequalityMatrix(DMatrixRMaj inequalityMatrix, DMatrixRMaj mutatedInequalityMatrix)
   {
      for (int i = 0; i < inequalityMatrix.getNumRows(); i++)
      {
         for (int j = 0; j < inequalityMatrix.getNumCols(); j++)
         {
            boolean mutate = random.nextInt(2) == 0;
            if (!mutate)
               continue;

            double val = inequalityMatrix.get(i, j);
            double mutationMultiplier = 1.0 + EuclidCoreRandomTools.nextDouble(random, 0.05);
            mutatedInequalityMatrix.set(i, j, val * mutationMultiplier);
         }
      }
   }

   private static DMatrixRMaj generateRandomConstraintMatrixVariation(int numRows, int numCols)
   {
      DMatrixRMaj constraintMatrixVariation = new DMatrixRMaj(numRows, numCols);

      for (int i = 0; i < numRows; i++)
      {
         for (int j = 0; j < numCols; j++)
         {
            constraintMatrixVariation.set(i, j, EuclidCoreRandomTools.nextDouble(random, 1.0));
         }
      }

      return constraintMatrixVariation;
   }

   private static boolean containsSameElements(TIntArrayList listA, TIntArrayList listB)
   {
      for (int i = 0; i < listA.size(); i++)
      {
         if (!listB.contains(listA.get(i)))
            return false;
      }
      return true;
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
            /* Assert that custom solver could not find solution when apache commons could not */
            Assertions.assertFalse(foundSimplexSolution);
            Assertions.assertFalse(foundCrissCrossSolution);
         }
         else
         {
            /* Assert that custom solver could find solution when apache commons could */
            Assertions.assertTrue(foundSimplexSolution);
            Assertions.assertTrue(foundCrissCrossSolution);

            /* Assert that solutions are equal */
            for (int j = 0; j < apacheCommonsSolution.length; j++)
            {
               Assertions.assertTrue(EuclidCoreTools.epsilonEquals(apacheCommonsSolution[j], simplexSolution.get(j), epsilon));
               Assertions.assertTrue(EuclidCoreTools.epsilonEquals(apacheCommonsSolution[j], crissCrossSolution.get(j), epsilon));
            }

            /* Check duality conditions are met */
            double primalObjective = 0.0;
            double dualObjective = 0.0;

            DMatrixRMaj b = constraintSet.equalityMatrix.getNumRows() == 0 ? constraintSet.inequalityVector : customSolver.getAugmentedInequalityVector();
            DMatrixRMaj dualSolution = customSolver.getDualSolution();

            for (int j = 0; j < costVector.getNumRows(); j++)
            {
               primalObjective += costVector.get(j) * simplexSolution.get(j);
            }
            for (int j = 0; j < b.getNumRows(); j++)
            {
               dualObjective += b.get(j) * dualSolution.get(j);
            }

            Assertions.assertTrue(Math.abs(primalObjective - dualObjective) < 1.0e-6);
         }
      }
   }

   @Test
   public void testComputeSensitivityToyProblem()
   {
      DMatrixRMaj Ain = new DMatrixRMaj(3, 2);
      Ain.set(0, 0, 1.0);
      Ain.set(1, 1, 1.0);
      Ain.set(2, 0, 1.0);
      Ain.set(2, 1, 1.0);

      DMatrixRMaj b = new DMatrixRMaj(3, 1);
      b.set(0, 0, 2.0);
      b.set(1, 0, 2.0);
      b.set(2, 0, 3.0);

      DMatrixRMaj c = new DMatrixRMaj(2, 1);
      c.set(0, 0, 2.0);
      c.set(1, 0, 1.0);

      DMatrixRMaj solution = new DMatrixRMaj(0);

      LinearProgramSolver solver = new LinearProgramSolver();
      solver.solve(c, Ain, b, solution);

      { // test variation of the constraint that is not in the active set, should have zero sensitivity
         DMatrixRMaj constraintVariation = new DMatrixRMaj(3, 2);
         constraintVariation.set(1, 0, 1.0);
         constraintVariation.set(1, 1, 1.0);
         double sensitivity = solver.computeSensitivity(constraintVariation);
         Assertions.assertTrue(Math.abs(sensitivity) < 1e-10, "Expected zero sensitivity for non-active constraints");
      }

      { // test variation of the constraint that is in the active set. compare to direct calculation
         DMatrixRMaj constraintVariation = new DMatrixRMaj(3, 2);

         Random random = new Random(3290);
         constraintVariation.set(0, 0, EuclidCoreRandomTools.nextDouble(random, 1.0));
         constraintVariation.set(0, 1, EuclidCoreRandomTools.nextDouble(random, 1.0));
         constraintVariation.set(2, 0, EuclidCoreRandomTools.nextDouble(random, 1.0));
         constraintVariation.set(2, 1, EuclidCoreRandomTools.nextDouble(random, 1.0));

         double expectedSensitivity = solver.computeSensitivity(constraintVariation);

         // direct calculation
         DMatrixRMaj z0 = new DMatrixRMaj(0);
         CommonOps_DDRM.multTransA(c, solution, z0);

         double theta = 1e-6;
         DMatrixRMaj constraintModification = new DMatrixRMaj(constraintVariation);
         CommonOps_DDRM.scale(theta, constraintModification);

         DMatrixRMaj modifiedConstraintMatrix = new DMatrixRMaj(Ain);
         CommonOps_DDRM.addEquals(modifiedConstraintMatrix, constraintModification);

         LinearProgramSolver directSensitivitySolver = new LinearProgramSolver();
         DMatrixRMaj directSensitivitySolution = new DMatrixRMaj(0);
         directSensitivitySolver.solve(c, modifiedConstraintMatrix, b, directSensitivitySolution);

         DMatrixRMaj zi = new DMatrixRMaj(0);
         CommonOps_DDRM.multTransA(c, directSensitivitySolution, zi);

         double computedSensitivity = (zi.get(0) - z0.get(0)) / theta;
         Assertions.assertTrue(Math.abs(expectedSensitivity - computedSensitivity) < 1e-6, "Expected and computed sensitivity do not match.");
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
