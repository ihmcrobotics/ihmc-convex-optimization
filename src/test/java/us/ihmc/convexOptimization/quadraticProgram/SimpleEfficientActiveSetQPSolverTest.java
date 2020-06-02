package us.ihmc.convexOptimization.quadraticProgram;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.ejml.data.DenseMatrix64F;
import org.junit.jupiter.api.Test;

import us.ihmc.matrixlib.MatrixTools;

public class SimpleEfficientActiveSetQPSolverTest extends AbstractSimpleActiveSetQPSolverTest
{
   @Override
   @Test
   public void testMaxIterations()
   {
      testMaxIterations(2, true);
   }

   @Override
   @Test
   public void testClear()
   {
      testClear(2, 2, false);
   }

   @Override
   @Test
   public void testSimpleCasesWithBoundsConstraints()
   {
      testSimpleCasesWithBoundsConstraints(1, 3, 2, 2, false);
   }

   @Override
   public DenseMatrix64F getLowerBounds()
   {
      // Need to modify the bounds for some tests to get a valid problem for this type of solver.
      return MatrixTools.createVector(-5.0, 6.0, 0.0);
   }

   @Override
   public ActiveSetQPSolver createSolverToTest()
   {
      SimpleEfficientActiveSetQPSolver simpleEfficientActiveSetQPSolver = new SimpleEfficientActiveSetQPSolver();
      simpleEfficientActiveSetQPSolver.setUseWarmStart(false);
      return simpleEfficientActiveSetQPSolver;

   }

   @Test
   public void testChallengingCasesWithPolygonConstraintsCheckFailsWithSimpleSolverWithWarmStart()
   {
      ActiveSetQPSolver solver = createSolverToTest();
      solver.setMaxNumberOfIterations(10);
      solver.setUseWarmStart(true);

      // Minimize x^2 + y^2 subject to x + y >= 2 (-x -y <= -2), y <= 10x - 2 (-10x + y <= -2), x <= 10y - 2 (x - 10y <= -2),
      // Equality solution will violate all three constraints, but optimal only has the first constraint active.
      // However, if you set all three constraints active, there is no solution.
      DenseMatrix64F costQuadraticMatrix = new DenseMatrix64F(new double[][] {{2.0, 0.0}, {0.0, 2.0}});
      DenseMatrix64F costLinearVector = MatrixTools.createVector(0.0, 0.0);
      double quadraticCostScalar = 0.0;
      solver.setQuadraticCostFunction(costQuadraticMatrix, costLinearVector, quadraticCostScalar);

      DenseMatrix64F linearInequalityConstraintsCMatrix = new DenseMatrix64F(new double[][] {{-1.0, -1.0}, {-10.0, 1.0}, {1.0, -10.0}});
      DenseMatrix64F linearInqualityConstraintsDVector = MatrixTools.createVector(-2.0, -2.0, -2.0);
      solver.setLinearInequalityConstraints(linearInequalityConstraintsCMatrix, linearInqualityConstraintsDVector);

      DenseMatrix64F solution = new DenseMatrix64F(2, 1);
      DenseMatrix64F lagrangeEqualityMultipliers = new DenseMatrix64F(0, 1);
      DenseMatrix64F lagrangeInequalityMultipliers = new DenseMatrix64F(3, 1);
      solver.solve(solution, lagrangeEqualityMultipliers, lagrangeInequalityMultipliers);
      int numberOfIterations = solver.solve(solution, lagrangeEqualityMultipliers, lagrangeInequalityMultipliers);

      assertEquals(2, solution.getNumRows());
      assertTrue(Double.isNaN(solution.get(0)));
      assertTrue(Double.isNaN(solution.get(1)));
      assertTrue(Double.isInfinite(lagrangeInequalityMultipliers.get(0)) || Double.isNaN(lagrangeInequalityMultipliers.get(0)));
      assertTrue(Double.isInfinite(lagrangeInequalityMultipliers.get(1)) || Double.isNaN(lagrangeInequalityMultipliers.get(1)));
      assertTrue(Double.isInfinite(lagrangeInequalityMultipliers.get(2)) || Double.isNaN(lagrangeInequalityMultipliers.get(2)));

      assertEquals(numberOfIterations, 1);
   }

   /**
    * Test with dataset from sim that revealed a bug with the variable lower/upper bounds handling.
    */
   @Test
   public void testFindValidSolutionForDataset20160319WithWarmStart()
   {
      ActualDatasetFrom20160319 dataset = new ActualDatasetFrom20160319();
      ActiveSetQPSolver solver = createSolverToTest();
      solver.setUseWarmStart(true);

      solver.clear();
      solver.setQuadraticCostFunction(dataset.getCostQuadraticMatrix(), dataset.getCostLinearVector(), 0.0);
      solver.setVariableBounds(dataset.getVariableLowerBounds(), dataset.getVariableUpperBounds());
      DenseMatrix64F solution = new DenseMatrix64F(dataset.getProblemSize(), 1);
      solver.solve(solution);
      int numberOfIterations = solver.solve(solution);

      assertFalse(MatrixTools.containsNaN(solution));
      assertEquals(numberOfIterations, 1);
   }

   @Test
   public void testFindValidSolutionForKiwiDataset20170712WithWarmStart()
   {
      ActualDatasetFromKiwi20170712 dataset = new ActualDatasetFromKiwi20170712();
      ActiveSetQPSolver solver = createSolverToTest();
      solver.setUseWarmStart(true);

      solver.clear();
      solver.setQuadraticCostFunction(dataset.getCostQuadraticMatrix(), dataset.getCostLinearVector(), 0.0);
      solver.setVariableBounds(dataset.getVariableLowerBounds(), dataset.getVariableUpperBounds());
      DenseMatrix64F solution = new DenseMatrix64F(dataset.getProblemSize(), 1);
      solver.solve(solution);
      int numberOfIterations = solver.solve(solution);

      assertFalse(MatrixTools.containsNaN(solution));
      assertEquals(numberOfIterations, 1);
   }

   /**
    * Test with dataset of a Kiwi simulation walking backward. It seems that the problem is related to
    * the fact that the robot has 6 contact points per foot. The solver still fails when increasing the
    * max number of iterations.
    */
   @Test
   public void testFindValidSolutionForKiwiDataset20171013WithWarmStart()
   {
      ActualDatasetFromKiwi20171013 dataset = new ActualDatasetFromKiwi20171013();
      ActiveSetQPSolver solver = createSolverToTest();
      solver.setUseWarmStart(true);

      solver.clear();
      solver.setQuadraticCostFunction(dataset.getCostQuadraticMatrix(), dataset.getCostLinearVector(), 0.0);
      solver.setVariableBounds(dataset.getVariableLowerBounds(), dataset.getVariableUpperBounds());
      DenseMatrix64F solution = new DenseMatrix64F(dataset.getProblemSize(), 1);
      solver.solve(solution);
      int numberOfIterations = solver.solve(solution);

      assertFalse(MatrixTools.containsNaN(solution));
      assertEquals(1, numberOfIterations);
   }

   @Test
   public void testFindValidSolutionForKiwiDatasetProblemWithWarmStart()
   {
      ActualDatasetFromKiwi20171015A datasetA = new ActualDatasetFromKiwi20171015A();
      ActualDatasetFromKiwi20171015B datasetB = new ActualDatasetFromKiwi20171015B();
      DenseMatrix64F solution = new DenseMatrix64F(datasetA.getProblemSize(), 1);

      ActiveSetQPSolver solver = createSolverToTest();
      solver.setUseWarmStart(true);

      solver.clear();
      solver.setQuadraticCostFunction(datasetA.getCostQuadraticMatrix(), datasetA.getCostLinearVector(), 0.0);
      solver.setVariableBounds(datasetA.getVariableLowerBounds(), datasetA.getVariableUpperBounds());
      solver.solve(solution);

      solver.clear();
      solver.setQuadraticCostFunction(datasetA.getCostQuadraticMatrix(), datasetA.getCostLinearVector(), 0.0);
      solver.setVariableBounds(datasetA.getVariableLowerBounds(), datasetA.getVariableUpperBounds());
      solver.solve(solution);

      solver.clear();
      solver.setQuadraticCostFunction(datasetB.getCostQuadraticMatrix(), datasetB.getCostLinearVector(), 0.0);
      solver.setVariableBounds(datasetB.getVariableLowerBounds(), datasetB.getVariableUpperBounds());
      int numberOfIterationsWithWarmStart = solver.solve(solution);

      assertFalse(MatrixTools.containsNaN(solution));
      //assertEquals(numberOfIterationsWithWarmStart, 1);

      solver.setUseWarmStart(false);
      solver.clear();
      solver.setQuadraticCostFunction(datasetB.getCostQuadraticMatrix(), datasetB.getCostLinearVector(), 0.0);
      solver.setVariableBounds(datasetB.getVariableLowerBounds(), datasetB.getVariableUpperBounds());
      int numberOfIterationsWithoutWarmStart = solver.solve(solution);

      assertTrue(numberOfIterationsWithWarmStart < numberOfIterationsWithoutWarmStart);
   }
}
