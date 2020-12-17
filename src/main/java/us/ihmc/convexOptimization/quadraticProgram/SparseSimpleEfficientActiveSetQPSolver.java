package us.ihmc.convexOptimization.quadraticProgram;

import gnu.trove.list.array.TIntArrayList;
import org.ejml.data.*;
import org.ejml.dense.row.CommonOps_DDRM;
import org.ejml.interfaces.linsol.LinearSolverSparse;
import org.ejml.sparse.FillReducing;
import org.ejml.sparse.csc.CommonOps_DSCC;
import org.ejml.sparse.csc.factory.LinearSolverFactory_DSCC;
import us.ihmc.convexOptimization.SparseMatrixTools;
import us.ihmc.log.LogTools;
import us.ihmc.matrixlib.MatrixTools;
import us.ihmc.matrixlib.NativeCommonOps;

/**
 * Solves a Quadratic Program using a simple active set method. Does not work for problems where
 * having multiple inequality constraints in the active set make the problem infeasible. Seems to
 * work well for problems with benign inequality constraints, such as variable bounds. Algorithm is
 * very fast when it can find a solution. Uses the algorithm and naming convention found in MIT
 * Paper "An efficiently solvable quadratic program for stabilizing dynamic locomotion" by Scott
 * Kuindersma, Frank Permenter, and Russ Tedrake.
 *
 * @author JerryPratt
 */
public class SparseSimpleEfficientActiveSetQPSolver implements ActiveSetQPSolver<DMatrixSparseCSC>
{
   private static final double violationFractionToAdd = 0.8;
   private static final double violationFractionToRemove = 0.95;
   //private static final double violationFractionToAdd = 1.0;
   //private static final double violationFractionToRemove = 1.0;
   private double convergenceThreshold = 1e-10;
   //private double convergenceThresholdForLagrangeMultipliers = 0.0;
   private double convergenceThresholdForLagrangeMultipliers = 1e-10;
   private int maxNumberOfIterations = 10;

   protected double quadraticCostScalar;

   private final DMatrixRMaj activeVariables = new DMatrixRMaj(0, 0);

   private final TIntArrayList activeInequalityIndices = new TIntArrayList();
   private final TIntArrayList activeUpperBoundIndices = new TIntArrayList();
   private final TIntArrayList activeLowerBoundIndices = new TIntArrayList();

   // Some temporary matrices:
   protected final DMatrixSparseCSC costQuadraticMatrix = new DMatrixSparseCSC(0, 0);
   protected final DMatrixSparseCSC costQuadraticMatrixTranspose = new DMatrixSparseCSC(0, 0);

   private final DMatrixRMaj linearInequalityConstraintsCheck = new DMatrixRMaj(0, 0);

   protected final DMatrixRMaj quadraticCostQVector = new DMatrixRMaj(0, 0);
   protected final DMatrixSparseCSC quadraticCostQMatrix = new DMatrixSparseCSC(0, 0);
   protected final DMatrixRMaj denseQuadraticCostQMatrix = new DMatrixRMaj(0, 0);
   protected final DMatrixSparseCSC linearEqualityConstraintsAMatrix = new DMatrixSparseCSC(0, 0);
   protected final DMatrixRMaj linearEqualityConstraintsBVector = new DMatrixRMaj(0, 0);
   protected final DMatrixRMaj tempLagrange = new DMatrixRMaj(0, 0);

   protected final DMatrixSparseCSC linearInequalityConstraintsCMatrixO = new DMatrixSparseCSC(0, 0);
   protected final DMatrixRMaj linearInequalityConstraintsDVectorO = new DMatrixRMaj(0, 0);

   protected final DMatrixRMaj variableLowerBounds = new DMatrixRMaj(0, 0);
   protected final DMatrixRMaj variableUpperBounds = new DMatrixRMaj(0, 0);

   private InverseCostCalculator<DMatrixSparseCSC> inverseSolver = new SparseInverseCostCalculator();
   private final LinearSolverSparse<DMatrixSparseCSC, DMatrixRMaj> lagrangeSolver = LinearSolverFactory_DSCC.lu(FillReducing.NONE);
   /**
    * Active inequality constraints
    */
   private final DMatrixSparseCSC CBar = new DMatrixSparseCSC(0, 0);
   private final DMatrixRMaj DBar = new DMatrixRMaj(0, 0);
   /**
    * Active variable bounds constraints
    */
   private final DMatrixSparseCSC CHat = new DMatrixSparseCSC(0, 0);
   private final DMatrixRMaj DHat = new DMatrixRMaj(0, 0);

   private final DMatrixSparseCSC QInverse = new DMatrixSparseCSC(0, 0);
   private final DMatrixSparseCSC AQInverse = new DMatrixSparseCSC(0, 0);
   private final DMatrixSparseCSC QInverseATranspose = new DMatrixSparseCSC(0, 0);
   private final DMatrixSparseCSC CBarQInverse = new DMatrixSparseCSC(0, 0);
   private final DMatrixSparseCSC CHatQInverse = new DMatrixSparseCSC(0, 0);
   private final DMatrixSparseCSC AQInverseATranspose = new DMatrixSparseCSC(0, 0);
   private final DMatrixSparseCSC AQInverseCBarTranspose = new DMatrixSparseCSC(0, 0);
   private final DMatrixSparseCSC AQInverseCHatTranspose = new DMatrixSparseCSC(0, 0);
   private final DMatrixSparseCSC CBarQInverseATranspose = new DMatrixSparseCSC(0, 0);
   private final DMatrixSparseCSC CHatQInverseATranspose = new DMatrixSparseCSC(0, 0);
   private final DMatrixSparseCSC QInverseCBarTranspose = new DMatrixSparseCSC(0, 0);
   private final DMatrixSparseCSC QInverseCHatTranspose = new DMatrixSparseCSC(0, 0);
   private final DMatrixSparseCSC CBarQInverseCBarTranspose = new DMatrixSparseCSC(0, 0);
   private final DMatrixSparseCSC CHatQInverseCHatTranspose = new DMatrixSparseCSC(0, 0);

   private final DMatrixSparseCSC CBarQInverseCHatTranspose = new DMatrixSparseCSC(0, 0);
   private final DMatrixSparseCSC CHatQInverseCBarTranspose = new DMatrixSparseCSC(0, 0);

   private final DMatrixSparseCSC AAndC = new DMatrixSparseCSC(0, 0);
   private final DMatrixRMaj ATransposeMuAndCTransposeLambda = new DMatrixRMaj(0, 0);

   private final DMatrixSparseCSC bigMatrixForLagrangeMultiplierSolution = new DMatrixSparseCSC(0, 0);
   private final DMatrixRMaj bigVectorForLagrangeMultiplierSolution = new DMatrixRMaj(0, 0);

   private final DMatrixRMaj tempVector = new DMatrixRMaj(0, 0);
   private final DMatrixRMaj augmentedLagrangeMultipliers = new DMatrixRMaj(0, 0);

   private final TIntArrayList inequalityIndicesToAddToActiveSet = new TIntArrayList();
   private final TIntArrayList inequalityIndicesToRemoveFromActiveSet = new TIntArrayList();

   private final TIntArrayList upperBoundIndicesToAddToActiveSet = new TIntArrayList();
   private final TIntArrayList upperBoundIndicesToRemoveFromActiveSet = new TIntArrayList();

   private final TIntArrayList lowerBoundIndicesToAddToActiveSet = new TIntArrayList();
   private final TIntArrayList lowerBoundIndicesToRemoveFromActiveSet = new TIntArrayList();

   protected final DMatrixRMaj computedObjectiveFunctionValue = new DMatrixRMaj(1, 1);

   private final DMatrixRMaj lowerBoundViolations = new DMatrixRMaj(0, 0);
   private final DMatrixRMaj upperBoundViolations = new DMatrixRMaj(0, 0);

   private boolean useWarmStart = false;

   private int previousNumberOfVariables = 0;
   private int previousNumberOfEqualityConstraints = 0;
   private int previousNumberOfInequalityConstraints = 0;
   private int previousNumberOfLowerBoundConstraints = 0;
   private int previousNumberOfUpperBoundConstraints = 0;

   private final IGrowArray gw = new IGrowArray();
   private final DGrowArray gx = new DGrowArray();

   @Override
   public void setConvergenceThreshold(double convergenceThreshold)
   {
      this.convergenceThreshold = convergenceThreshold;
   }

   @Override
   public void setMaxNumberOfIterations(int maxNumberOfIterations)
   {
      this.maxNumberOfIterations = maxNumberOfIterations;
   }

   @Override
   public void clear()
   {
      quadraticCostQMatrix.reshape(0, 0);
      denseQuadraticCostQMatrix.reshape(0, 0);
      quadraticCostQVector.reshape(0, 0);

      linearEqualityConstraintsAMatrix.reshape(0, 0);
      linearEqualityConstraintsBVector.reshape(0, 0);

      linearInequalityConstraintsCMatrixO.reshape(0, 0);
      linearInequalityConstraintsDVectorO.reshape(0, 0);

      variableLowerBounds.reshape(0, 0);
      variableUpperBounds.reshape(0, 0);

      lowerBoundViolations.reshape(0, 0);
      upperBoundViolations.reshape(0, 0);
   }

   @Override
   public void setLowerBounds(DMatrixRMaj variableLowerBounds)
   {
      if (variableLowerBounds.getNumRows() != quadraticCostQMatrix.getNumRows())
         throw new RuntimeException("variableLowerBounds.getNumRows() != quadraticCostQMatrix.getNumRows()");

      this.variableLowerBounds.set(variableLowerBounds);
   }

   @Override
   public void setUpperBounds(DMatrixRMaj variableUpperBounds)
   {
      if (variableUpperBounds.getNumRows() != quadraticCostQMatrix.getNumRows())
         throw new RuntimeException("variableUpperBounds.getNumRows() != quadraticCostQMatrix.getNumRows()");

      this.variableUpperBounds.set(variableUpperBounds);
   }

   @Override
   public void setQuadraticCostFunction(DMatrixSparseCSC H, DMatrixRMaj f)
   {
      setQuadraticCostFunction(H, f, 0.0);
   }

   @Override
   public void setQuadraticCostFunction(DMatrixSparseCSC costQuadraticMatrix, DMatrixRMaj costLinearVector, double quadraticCostScalar)
   {
      if (costLinearVector.getNumCols() != 1)
         throw new RuntimeException("costLinearVector.getNumCols() != 1");
      if (costQuadraticMatrix.getNumRows() != costLinearVector.getNumRows())
         throw new RuntimeException("costQuadraticMatrix.getNumRows() != costLinearVector.getNumRows()");
      if (costQuadraticMatrix.getNumRows() != costQuadraticMatrix.getNumCols())
         throw new RuntimeException("costQuadraticMatrix.getNumRows() != costQuadraticMatrix.getNumCols()");

      this.costQuadraticMatrix.set(costQuadraticMatrix);
      CommonOps_DSCC.transpose(this.costQuadraticMatrix, costQuadraticMatrixTranspose, gw);
      CommonOps_DSCC.add(0.5, this.costQuadraticMatrix, 0.5, costQuadraticMatrixTranspose, quadraticCostQMatrix, gw, gx);
      denseQuadraticCostQMatrix.set(quadraticCostQMatrix);

      quadraticCostQVector.set(costLinearVector);
      this.quadraticCostScalar = quadraticCostScalar;
   }

   @Override
   public double getObjectiveCost(DMatrixRMaj x)
   {
      NativeCommonOps.multQuad(x, denseQuadraticCostQMatrix, computedObjectiveFunctionValue);
      CommonOps_DDRM.scale(0.5, computedObjectiveFunctionValue);

      CommonOps_DDRM.multAddTransA(quadraticCostQVector, x, computedObjectiveFunctionValue);
      return computedObjectiveFunctionValue.get(0, 0) + quadraticCostScalar;
   }

   @Override
   public void setLinearEqualityConstraints(DMatrixSparseCSC linearEqualityConstraintsAMatrix, DMatrixRMaj linearEqualityConstraintsBVector)
   {
      if (linearEqualityConstraintsBVector.getNumCols() != 1)
         throw new RuntimeException("linearEqualityConstraintsBVector.getNumCols() != 1");
      if (linearEqualityConstraintsAMatrix.getNumRows() != linearEqualityConstraintsBVector.getNumRows())
         throw new RuntimeException("linearEqualityConstraintsAMatrix.getNumRows() != linearEqualityConstraintsBVector.getNumRows()");
      if (linearEqualityConstraintsAMatrix.getNumCols() != quadraticCostQMatrix.getNumCols())
         throw new RuntimeException("linearEqualityConstraintsAMatrix.getNumCols() != quadraticCostQMatrix.getNumCols()");

      this.linearEqualityConstraintsBVector.set(linearEqualityConstraintsBVector);
      this.linearEqualityConstraintsAMatrix.set(linearEqualityConstraintsAMatrix);
   }

   @Override
   public void setLinearInequalityConstraints(DMatrixSparseCSC linearInequalityConstraintCMatrix, DMatrixRMaj linearInequalityConstraintDVector)
   {
      if (linearInequalityConstraintDVector.getNumCols() != 1)
         throw new RuntimeException("linearInequalityConstraintDVector.getNumCols() != 1");
      if (linearInequalityConstraintCMatrix.getNumRows() != linearInequalityConstraintDVector.getNumRows())
         throw new RuntimeException("linearInequalityConstraintCMatrix.getNumRows() != linearInequalityConstraintDVector.getNumRows()");
      if (linearInequalityConstraintCMatrix.getNumCols() != quadraticCostQMatrix.getNumCols())
         throw new RuntimeException("linearInequalityConstraintCMatrix.getNumCols() != quadraticCostQMatrix.getNumCols()");

      linearInequalityConstraintsDVectorO.set(linearInequalityConstraintDVector);
      linearInequalityConstraintsCMatrixO.set(linearInequalityConstraintCMatrix);
   }

   @Override
   public void setUseWarmStart(boolean useWarmStart)
   {
      this.useWarmStart = useWarmStart;
   }

   @Override
   public void resetActiveSet()
   {
      CBar.reshape(0, 0);
      CHat.reshape(0, 0);
      DBar.reshape(0, 0);
      DHat.reshape(0, 0);

      activeInequalityIndices.reset();
      activeUpperBoundIndices.reset();
      activeLowerBoundIndices.reset();
   }

   private final DMatrixRMaj lagrangeEqualityConstraintMultipliers = new DMatrixRMaj(0, 0);
   private final DMatrixRMaj lagrangeInequalityConstraintMultipliers = new DMatrixRMaj(0, 0);
   private final DMatrixRMaj lagrangeLowerBoundMultipliers = new DMatrixRMaj(0, 0);
   private final DMatrixRMaj lagrangeUpperBoundMultipliers = new DMatrixRMaj(0, 0);

   @Override
   public int solve(DMatrixRMaj solutionToPack)
   {
      if (!useWarmStart || problemSizeChanged())
         resetActiveSet();
      else
         addActiveSetConstraintsAsEqualityConstraints();

      int numberOfIterations = 0;

      int numberOfVariables = quadraticCostQMatrix.getNumRows();
      int numberOfEqualityConstraints = linearEqualityConstraintsAMatrix.getNumRows();
      int numberOfInequalityConstraints = linearInequalityConstraintsCMatrixO.getNumRows();
      int numberOfLowerBoundConstraints = variableLowerBounds.getNumRows();
      int numberOfUpperBoundConstraints = variableUpperBounds.getNumRows();

      solutionToPack.reshape(numberOfVariables, 1);
      lagrangeEqualityConstraintMultipliers.reshape(numberOfEqualityConstraints, 1);
      lagrangeEqualityConstraintMultipliers.zero();
      lagrangeInequalityConstraintMultipliers.reshape(numberOfInequalityConstraints, 1);
      lagrangeInequalityConstraintMultipliers.zero();
      lagrangeLowerBoundMultipliers.reshape(numberOfLowerBoundConstraints, 1);
      lagrangeLowerBoundMultipliers.zero();
      lagrangeUpperBoundMultipliers.reshape(numberOfUpperBoundConstraints, 1);
      lagrangeUpperBoundMultipliers.zero();

      computeQInverseAndAQInverse();

      solveEqualityConstrainedSubproblemEfficiently(solutionToPack,
                                                    lagrangeEqualityConstraintMultipliers,
                                                    lagrangeInequalityConstraintMultipliers,
                                                    lagrangeLowerBoundMultipliers,
                                                    lagrangeUpperBoundMultipliers);

      //      System.out.println(numberOfInequalityConstraints + ", " + numberOfLowerBoundConstraints + ", " + numberOfUpperBoundConstraints);
      if (numberOfInequalityConstraints == 0 && numberOfLowerBoundConstraints == 0 && numberOfUpperBoundConstraints == 0)
      {
         return numberOfIterations;
      }

      for (int i = 0; i < maxNumberOfIterations; i++)
      {
         boolean activeSetWasModified = modifyActiveSetAndTryAgain(solutionToPack,
                                                                   lagrangeEqualityConstraintMultipliers,
                                                                   lagrangeInequalityConstraintMultipliers,
                                                                   lagrangeLowerBoundMultipliers,
                                                                   lagrangeUpperBoundMultipliers);
         numberOfIterations++;

         if (!activeSetWasModified)
         {
            return numberOfIterations;
         }
      }

      // No solution found. Pack NaN in all variables
      solutionToPack.reshape(numberOfVariables, 1);
      for (int i = 0; i < numberOfVariables; i++)
         solutionToPack.set(i, 0, Double.NaN);

      return numberOfIterations;
   }

   private boolean problemSizeChanged()
   {
      boolean sizeChanged = checkProblemSize();

      previousNumberOfVariables = (int) CommonOps_DDRM.elementSum(activeVariables);
      previousNumberOfEqualityConstraints = linearEqualityConstraintsAMatrix.getNumRows();
      previousNumberOfInequalityConstraints = linearInequalityConstraintsCMatrixO.getNumRows();
      previousNumberOfLowerBoundConstraints = variableLowerBounds.getNumRows();
      previousNumberOfUpperBoundConstraints = variableUpperBounds.getNumRows();

      return sizeChanged;
   }

   private boolean checkProblemSize()
   {
      if (previousNumberOfVariables != CommonOps_DDRM.elementSum(activeVariables))
         return true;
      if (previousNumberOfEqualityConstraints != linearEqualityConstraintsAMatrix.getNumRows())
         return true;
      if (previousNumberOfInequalityConstraints != linearInequalityConstraintsCMatrixO.getNumRows())
         return true;
      if (previousNumberOfLowerBoundConstraints != variableLowerBounds.getNumRows())
         return true;
      if (previousNumberOfUpperBoundConstraints != variableUpperBounds.getNumRows())
         return true;

      return false;
   }


   private void computeQInverseAndAQInverse()
   {
      int numberOfVariables = quadraticCostQMatrix.getNumRows();
      int numberOfEqualityConstraints = linearEqualityConstraintsAMatrix.getNumRows();

      inverseSolver.computeInverse(quadraticCostQMatrix, QInverse);

      if (numberOfEqualityConstraints > 0)
      {
         CommonOps_DSCC.mult(linearEqualityConstraintsAMatrix, QInverse, AQInverse, gw, gx);
         CommonOps_DSCC.multTransB(QInverse, linearEqualityConstraintsAMatrix, QInverseATranspose, gw, gx);
         CommonOps_DSCC.multTransB(AQInverse, linearEqualityConstraintsAMatrix, AQInverseATranspose, gw, gx);
      }
      else
      {
         AQInverse.reshape(numberOfEqualityConstraints, numberOfVariables);
         QInverseATranspose.reshape(numberOfVariables, numberOfEqualityConstraints);
         AQInverseATranspose.reshape(numberOfEqualityConstraints, numberOfEqualityConstraints);
      }
   }

   private void computeCBarTempMatrices()
   {
      if (CBar.getNumRows() > 0)
      {
         CommonOps_DSCC.multTransB(AQInverse, CBar, AQInverseCBarTranspose, gw, gx);

         CommonOps_DSCC.mult(CBar, QInverseATranspose, CBarQInverseATranspose, gw, gx);

         CommonOps_DSCC.mult(CBar, QInverse, CBarQInverse, gw, gx);

         CommonOps_DSCC.multTransB(QInverse, CBar, QInverseCBarTranspose, gw, gx);

         CommonOps_DSCC.mult(CBar, QInverseCBarTranspose, CBarQInverseCBarTranspose, gw, gx);
      }
      else
      {
         AQInverseCBarTranspose.reshape(0, 0);
         CBarQInverseATranspose.reshape(0, 0);
         CBarQInverse.reshape(0, 0);
         QInverseCBarTranspose.reshape(0, 0);
         CBarQInverseCBarTranspose.reshape(0, 0);
      }
   }

   private void computeCHatTempMatrices()
   {
      if (CHat.getNumRows() > 0)
      {
         CommonOps_DSCC.multTransB(AQInverse, CHat, AQInverseCHatTranspose, gw, gx);

         CommonOps_DSCC.mult(CHat, QInverseATranspose, CHatQInverseATranspose, gw, gx);

         CommonOps_DSCC.mult(CHat, QInverse, CHatQInverse, gw, gx);

         CommonOps_DSCC.multTransB(QInverse, CHat, QInverseCHatTranspose, gw, gx);

         CommonOps_DSCC.mult(CHat, QInverseCHatTranspose, CHatQInverseCHatTranspose, gw, gx);

         if (CBar.getNumRows() > 0)
         {
            CommonOps_DSCC.mult(CBar, QInverseCHatTranspose, CBarQInverseCHatTranspose, gw, gx);
            CommonOps_DSCC.mult(CHat, QInverseCBarTranspose, CHatQInverseCBarTranspose, gw, gx);
         }
         else
         {
            CBarQInverseCHatTranspose.reshape(CBar.getNumRows(), CHat.getNumRows());
            CHatQInverseCBarTranspose.reshape(CHat.getNumRows(), CBar.getNumRows());
         }
      }
      else
      {
         AQInverseCHatTranspose.reshape(0, 0);
         CHatQInverseATranspose.reshape(0, 0);
         CHatQInverse.reshape(0, 0);
         QInverseCHatTranspose.reshape(0, 0);
         CHatQInverseCHatTranspose.reshape(0, 0);
         CBarQInverseCHatTranspose.reshape(0, 0);
         CHatQInverseCBarTranspose.reshape(0, 0);
      }
   }

   private boolean modifyActiveSetAndTryAgain(DMatrixRMaj solutionToPack,
                                              DMatrixRMaj lagrangeEqualityConstraintMultipliersToPack,
                                              DMatrixRMaj lagrangeInequalityConstraintMultipliersToPack,
                                              DMatrixRMaj lagrangeLowerBoundConstraintMultipliersToPack,
                                              DMatrixRMaj lagrangeUpperBoundConstraintMultipliersToPack)
   {
      if (MatrixTools.containsNaN(solutionToPack))
         return false;

      boolean activeSetWasModified = false;

      // find the constraints to add
      int numberOfInequalityConstraints = linearInequalityConstraintsCMatrixO.getNumRows();
      int numberOfLowerBoundConstraints = variableLowerBounds.getNumRows();
      int numberOfUpperBoundConstraints = variableUpperBounds.getNumRows();

      double maxInequalityViolation = Double.NEGATIVE_INFINITY, maxLowerBoundViolation = Double.NEGATIVE_INFINITY, maxUpperBoundViolation = Double.NEGATIVE_INFINITY;
      if (numberOfInequalityConstraints != 0)
      {
         CommonOps_DDRM.scale(-1.0, linearInequalityConstraintsDVectorO, linearInequalityConstraintsCheck);
         CommonOps_DSCC.multAdd(linearInequalityConstraintsCMatrixO, solutionToPack, linearInequalityConstraintsCheck);

         for (int i = 0; i < linearInequalityConstraintsCheck.getNumRows(); i++)
         {
            if (activeInequalityIndices.contains(i))
               continue;

            if (linearInequalityConstraintsCheck.get(i, 0) >= maxInequalityViolation)
               maxInequalityViolation = linearInequalityConstraintsCheck.get(i, 0);
         }
      }

      if (numberOfLowerBoundConstraints != 0)
      {
         CommonOps_DDRM.subtract(variableLowerBounds, solutionToPack, lowerBoundViolations);

         for (int i = 0; i < lowerBoundViolations.getNumRows(); i++)
         {
            if (activeLowerBoundIndices.contains(i))
               continue;

            if (lowerBoundViolations.get(i, 0) >= maxLowerBoundViolation)
               maxLowerBoundViolation = lowerBoundViolations.get(i, 0);
         }
      }
      else
      {
         lowerBoundViolations.reshape(numberOfLowerBoundConstraints, 1);
      }

      if (numberOfUpperBoundConstraints != 0)
      {
         CommonOps_DDRM.subtract(solutionToPack, variableUpperBounds, upperBoundViolations);

         for (int i = 0; i < upperBoundViolations.getNumRows(); i++)
         {
            if (activeUpperBoundIndices.contains(i))
               continue;

            if (upperBoundViolations.get(i, 0) >= maxUpperBoundViolation)
               maxUpperBoundViolation = upperBoundViolations.get(i, 0);
         }
      }
      else
      {
         upperBoundViolations.reshape(numberOfUpperBoundConstraints, 1);
      }

      double maxConstraintViolation = Math.max(maxInequalityViolation, Math.max(maxLowerBoundViolation, maxUpperBoundViolation));
      double minViolationToAdd = (1.0 - violationFractionToAdd) * maxConstraintViolation + convergenceThreshold;

      // check inequality constraints
      inequalityIndicesToAddToActiveSet.reset();
      if (maxInequalityViolation > minViolationToAdd)
      {
         for (int i = 0; i < numberOfInequalityConstraints; i++)
         {
            if (activeInequalityIndices.contains(i))
               continue; // Only check violation on those that are not active. Otherwise check should just return 0.0, but roundoff could cause problems.

            if (linearInequalityConstraintsCheck.get(i, 0) > minViolationToAdd)
            {
               activeSetWasModified = true;
               inequalityIndicesToAddToActiveSet.add(i);
            }
         }
      }

      // Check the lower bounds
      lowerBoundIndicesToAddToActiveSet.reset();
      if (maxLowerBoundViolation > minViolationToAdd)
      {
         for (int i = 0; i < numberOfLowerBoundConstraints; i++)
         {
            if (activeLowerBoundIndices.contains(i))
               continue; // Only check violation on those that are not active. Otherwise check should just return 0.0, but roundoff could cause problems.

            if (lowerBoundViolations.get(i, 0) > minViolationToAdd)
            {
               activeSetWasModified = true;
               lowerBoundIndicesToAddToActiveSet.add(i);
            }
         }
      }

      // Check the upper bounds
      upperBoundIndicesToAddToActiveSet.reset();
      if (maxUpperBoundViolation > minViolationToAdd)
      {
         for (int i = 0; i < numberOfUpperBoundConstraints; i++)
         {
            if (activeUpperBoundIndices.contains(i))
               continue; // Only check violation on those that are not active. Otherwise check should just return 0.0, but roundoff could cause problems.

            if (upperBoundViolations.get(i, 0) > minViolationToAdd)
            {
               activeSetWasModified = true;
               upperBoundIndicesToAddToActiveSet.add(i);
            }
         }
      }

      // find the constraints to remove
      int numberOfActiveInequalityConstraints = activeInequalityIndices.size();
      int numberOfActiveUpperBounds = activeUpperBoundIndices.size();
      int numberOfActiveLowerBounds = activeLowerBoundIndices.size();

      double minLagrangeInequalityMultiplier = Double.POSITIVE_INFINITY, minLagrangeLowerBoundMultiplier = Double.POSITIVE_INFINITY, minLagrangeUpperBoundMultiplier = Double.POSITIVE_INFINITY;

      if (numberOfActiveInequalityConstraints != 0)
         minLagrangeInequalityMultiplier = CommonOps_DDRM.elementMin(lagrangeInequalityConstraintMultipliersToPack);
      if (numberOfActiveLowerBounds != 0)
         minLagrangeLowerBoundMultiplier = CommonOps_DDRM.elementMin(lagrangeLowerBoundConstraintMultipliersToPack);
      if (numberOfActiveUpperBounds != 0)
         minLagrangeUpperBoundMultiplier = CommonOps_DDRM.elementMin(lagrangeUpperBoundConstraintMultipliersToPack);

      double minLagrangeMultiplier = Math.min(minLagrangeInequalityMultiplier, Math.min(minLagrangeLowerBoundMultiplier, minLagrangeUpperBoundMultiplier));
      double maxLagrangeMultiplierToRemove = -(1.0 - violationFractionToRemove) * minLagrangeMultiplier - convergenceThresholdForLagrangeMultipliers;

      inequalityIndicesToRemoveFromActiveSet.reset();
      if (minLagrangeInequalityMultiplier < maxLagrangeMultiplierToRemove)
      {
         for (int i = 0; i < activeInequalityIndices.size(); i++)
         {
            int indexToCheck = activeInequalityIndices.get(i);

            double lagrangeMultiplier = lagrangeInequalityConstraintMultipliersToPack.get(indexToCheck, 0);
            if (lagrangeMultiplier < maxLagrangeMultiplierToRemove)
            {
               activeSetWasModified = true;
               inequalityIndicesToRemoveFromActiveSet.add(indexToCheck);
            }
         }
      }

      lowerBoundIndicesToRemoveFromActiveSet.reset();
      if (minLagrangeLowerBoundMultiplier < maxLagrangeMultiplierToRemove)
      {
         for (int i = 0; i < activeLowerBoundIndices.size(); i++)
         {
            int indexToCheck = activeLowerBoundIndices.get(i);

            double lagrangeMultiplier = lagrangeLowerBoundConstraintMultipliersToPack.get(indexToCheck, 0);
            if (lagrangeMultiplier < maxLagrangeMultiplierToRemove)
            {
               activeSetWasModified = true;
               lowerBoundIndicesToRemoveFromActiveSet.add(indexToCheck);
            }
         }
      }

      upperBoundIndicesToRemoveFromActiveSet.reset();
      if (minLagrangeUpperBoundMultiplier < maxLagrangeMultiplierToRemove)
      {
         for (int i = 0; i < activeUpperBoundIndices.size(); i++)
         {
            int indexToCheck = activeUpperBoundIndices.get(i);

            double lagrangeMultiplier = lagrangeUpperBoundConstraintMultipliersToPack.get(indexToCheck, 0);
            if (lagrangeMultiplier < maxLagrangeMultiplierToRemove)
            {
               activeSetWasModified = true;
               upperBoundIndicesToRemoveFromActiveSet.add(indexToCheck);
            }
         }
      }

      if (!activeSetWasModified)
         return false;

      for (int i = 0; i < inequalityIndicesToAddToActiveSet.size(); i++)
      {
         activeInequalityIndices.add(inequalityIndicesToAddToActiveSet.get(i));
      }
      for (int i = 0; i < inequalityIndicesToRemoveFromActiveSet.size(); i++)
      {
         activeInequalityIndices.remove(inequalityIndicesToRemoveFromActiveSet.get(i));
      }

      for (int i = 0; i < lowerBoundIndicesToAddToActiveSet.size(); i++)
      {
         activeLowerBoundIndices.add(lowerBoundIndicesToAddToActiveSet.get(i));
      }
      for (int i = 0; i < lowerBoundIndicesToRemoveFromActiveSet.size(); i++)
      {
         activeLowerBoundIndices.remove(lowerBoundIndicesToRemoveFromActiveSet.get(i));
      }

      for (int i = 0; i < upperBoundIndicesToAddToActiveSet.size(); i++)
      {
         activeUpperBoundIndices.add(upperBoundIndicesToAddToActiveSet.get(i));
      }
      for (int i = 0; i < upperBoundIndicesToRemoveFromActiveSet.size(); i++)
      {
         activeUpperBoundIndices.remove(upperBoundIndicesToRemoveFromActiveSet.get(i));
      }

      // Add active set constraints as equality constraints:
      addActiveSetConstraintsAsEqualityConstraints();

      solveEqualityConstrainedSubproblemEfficiently(solutionToPack,
                                                    lagrangeEqualityConstraintMultipliersToPack,
                                                    lagrangeInequalityConstraintMultipliersToPack,
                                                    lagrangeLowerBoundConstraintMultipliersToPack,
                                                    lagrangeUpperBoundConstraintMultipliersToPack);

      return true;
   }

   private void addActiveSetConstraintsAsEqualityConstraints()
   {
      int numberOfVariables = quadraticCostQMatrix.getNumRows();

      int sizeOfActiveSet = activeInequalityIndices.size();

      CBar.reshape(sizeOfActiveSet, numberOfVariables);
      DBar.reshape(sizeOfActiveSet, 1);

      for (int i = 0; i < sizeOfActiveSet; i++)
      {
         int inequalityConstraintIndex = activeInequalityIndices.get(i);
         // todo make this column-wise, rather than row-wise
         CommonOps_DSCC.extract(linearInequalityConstraintsCMatrixO, inequalityConstraintIndex, inequalityConstraintIndex + 1, 0, numberOfVariables, CBar, i, 0);
         CommonOps_DDRM.extract(linearInequalityConstraintsDVectorO, inequalityConstraintIndex, inequalityConstraintIndex + 1, 0, 1, DBar, i, 0);
      }

      // Add active bounds constraints as equality constraints:
      int sizeOfLowerBoundsActiveSet = activeLowerBoundIndices.size();
      int sizeOfUpperBoundsActiveSet = activeUpperBoundIndices.size();

      int sizeOfBoundsActiveSet = sizeOfLowerBoundsActiveSet + sizeOfUpperBoundsActiveSet;

      CHat.reshape(sizeOfBoundsActiveSet, numberOfVariables);
      DHat.reshape(sizeOfBoundsActiveSet, 1);

      CHat.zero();
      DHat.zero();

      int row = 0;

      for (int i = 0; i < sizeOfLowerBoundsActiveSet; i++)
      {
         int lowerBoundsConstraintIndex = activeLowerBoundIndices.get(i);

         CHat.set(row, lowerBoundsConstraintIndex, -1.0);
         DHat.set(row, 0, -variableLowerBounds.get(lowerBoundsConstraintIndex, 0));
         row++;
      }

      for (int i = 0; i < sizeOfUpperBoundsActiveSet; i++)
      {
         int upperBoundsConstraintIndex = activeUpperBoundIndices.get(i);

         CHat.set(row, upperBoundsConstraintIndex, 1.0);
         DHat.set(row, 0, variableUpperBounds.get(upperBoundsConstraintIndex, 0));
         row++;
      }

      //printSetChanges();
   }

   private void printSetChanges()
   {
      if (!lowerBoundIndicesToAddToActiveSet.isEmpty())
      {
         LogTools.info("Lower bound indices added : ");
         for (int i = 0; i < lowerBoundIndicesToAddToActiveSet.size(); i++)
            LogTools.info("" + lowerBoundIndicesToAddToActiveSet.get(i));
      }
      if (!lowerBoundIndicesToRemoveFromActiveSet.isEmpty())
      {
         LogTools.info("Lower bound indices removed : ");
         for (int i = 0; i < lowerBoundIndicesToRemoveFromActiveSet.size(); i++)
            LogTools.info("" + lowerBoundIndicesToRemoveFromActiveSet.get(i));
      }

      if (!upperBoundIndicesToAddToActiveSet.isEmpty())
      {
         LogTools.info("Upper bound indices added : ");
         for (int i = 0; i < upperBoundIndicesToAddToActiveSet.size(); i++)
            LogTools.info("" + upperBoundIndicesToAddToActiveSet.get(i));
      }
      if (!upperBoundIndicesToRemoveFromActiveSet.isEmpty())
      {
         LogTools.info("Upper bound indices removed : ");
         for (int i = 0; i < upperBoundIndicesToRemoveFromActiveSet.size(); i++)
            LogTools.info("" + upperBoundIndicesToRemoveFromActiveSet.get(i));
      }

      if (!inequalityIndicesToAddToActiveSet.isEmpty())
      {
         LogTools.info("Inequality constraint indices added : ");
         for (int i = 0; i < inequalityIndicesToAddToActiveSet.size(); i++)
            LogTools.info("" + inequalityIndicesToAddToActiveSet.get(i));
      }
      if (!inequalityIndicesToRemoveFromActiveSet.isEmpty())
      {
         LogTools.info("Inequality constraint indices removed : ");
         for (int i = 0; i < inequalityIndicesToRemoveFromActiveSet.size(); i++)
            LogTools.info("" + inequalityIndicesToRemoveFromActiveSet.get(i));
      }
   }

   private void solveEqualityConstrainedSubproblemEfficiently(DMatrixRMaj xSolutionToPack,
                                                              DMatrixRMaj lagrangeEqualityConstraintMultipliersToPack,
                                                              DMatrixRMaj lagrangeInequalityConstraintMultipliersToPack,
                                                              DMatrixRMaj lagrangeLowerBoundConstraintMultipliersToPack,
                                                              DMatrixRMaj lagrangeUpperBoundConstraintMultipliersToPack)
   {
      int numberOfVariables = quadraticCostQMatrix.getNumRows();
      int numberOfOriginalEqualityConstraints = linearEqualityConstraintsAMatrix.getNumRows();

      int numberOfActiveInequalityConstraints = activeInequalityIndices.size();
      int numberOfActiveLowerBoundConstraints = activeLowerBoundIndices.size();
      int numberOfActiveUpperBoundConstraints = activeUpperBoundIndices.size();

      int numberOfAugmentedEqualityConstraints = numberOfOriginalEqualityConstraints + numberOfActiveInequalityConstraints + numberOfActiveLowerBoundConstraints
                                                 + numberOfActiveUpperBoundConstraints;

      if (numberOfAugmentedEqualityConstraints == 0)
      {
         CommonOps_DSCC.mult(QInverse, quadraticCostQVector, xSolutionToPack);
         CommonOps_DDRM.scale(-1.0, xSolutionToPack);
         return;
      }

      computeCBarTempMatrices();
      computeCHatTempMatrices();

      augmentedLagrangeMultipliers.reshape(numberOfAugmentedEqualityConstraints, 1);
      bigMatrixForLagrangeMultiplierSolution.reshape(numberOfAugmentedEqualityConstraints, numberOfAugmentedEqualityConstraints);
      bigVectorForLagrangeMultiplierSolution.reshape(numberOfAugmentedEqualityConstraints, 1);

      // TODO size the bigMatrixForLagrange... and make a custom method to pack things in vertically, which is way more efficient
      SparseMatrixTools.insert(AQInverseATranspose, bigMatrixForLagrangeMultiplierSolution, 0, 0);
      SparseMatrixTools.insert(AQInverseCBarTranspose, bigMatrixForLagrangeMultiplierSolution, 0, numberOfOriginalEqualityConstraints);
      SparseMatrixTools.insert(AQInverseCHatTranspose,
                               bigMatrixForLagrangeMultiplierSolution,
                               0,
                               numberOfOriginalEqualityConstraints + numberOfActiveInequalityConstraints);

      SparseMatrixTools.insert(CBarQInverseATranspose, bigMatrixForLagrangeMultiplierSolution, numberOfOriginalEqualityConstraints, 0);
      SparseMatrixTools.insert(CBarQInverseCBarTranspose,
                               bigMatrixForLagrangeMultiplierSolution,
                               numberOfOriginalEqualityConstraints,
                               numberOfOriginalEqualityConstraints);
      SparseMatrixTools.insert(CBarQInverseCHatTranspose,
                               bigMatrixForLagrangeMultiplierSolution,
                               numberOfOriginalEqualityConstraints,
                               numberOfOriginalEqualityConstraints + numberOfActiveInequalityConstraints);
      SparseMatrixTools.insert(CHatQInverseATranspose,
                               bigMatrixForLagrangeMultiplierSolution,
                               numberOfOriginalEqualityConstraints + numberOfActiveInequalityConstraints,
                               0);
      SparseMatrixTools.insert(CHatQInverseCBarTranspose,
                               bigMatrixForLagrangeMultiplierSolution,
                               numberOfOriginalEqualityConstraints + numberOfActiveInequalityConstraints,
                               numberOfOriginalEqualityConstraints);
      SparseMatrixTools.insert(CHatQInverseCHatTranspose,
                               bigMatrixForLagrangeMultiplierSolution,
                               numberOfOriginalEqualityConstraints + numberOfActiveInequalityConstraints,
                               numberOfOriginalEqualityConstraints + numberOfActiveInequalityConstraints);

      if (numberOfOriginalEqualityConstraints > 0)
      {
         tempLagrange.set(linearEqualityConstraintsBVector);
         CommonOps_DSCC.multAdd(AQInverse, quadraticCostQVector, tempLagrange);
         CommonOps_DDRM.insert(tempLagrange, bigVectorForLagrangeMultiplierSolution, 0, 0);
      }

      if (numberOfActiveInequalityConstraints > 0)
      {
         tempLagrange.set(DBar);
         CommonOps_DSCC.multAdd(CBarQInverse, quadraticCostQVector, tempLagrange);
         CommonOps_DDRM.insert(tempLagrange, bigVectorForLagrangeMultiplierSolution, numberOfOriginalEqualityConstraints, 0);
      }

      if (numberOfActiveLowerBoundConstraints + numberOfActiveUpperBoundConstraints > 0)
      {
         tempLagrange.set(DHat);
         CommonOps_DSCC.multAdd(CHatQInverse, quadraticCostQVector, tempLagrange);
         CommonOps_DDRM.insert(tempLagrange, bigVectorForLagrangeMultiplierSolution, numberOfOriginalEqualityConstraints + numberOfActiveInequalityConstraints, 0);
      }

      CommonOps_DDRM.scale(-1.0, bigVectorForLagrangeMultiplierSolution);

      lagrangeSolver.setA(bigMatrixForLagrangeMultiplierSolution);
      lagrangeSolver.solve(bigVectorForLagrangeMultiplierSolution, augmentedLagrangeMultipliers);

      AAndC.reshape(numberOfAugmentedEqualityConstraints, numberOfVariables);
      SparseMatrixTools.insert(linearEqualityConstraintsAMatrix, AAndC, 0, 0);
      SparseMatrixTools.insert(CBar, AAndC, numberOfOriginalEqualityConstraints, 0);
      SparseMatrixTools.insert(CHat, AAndC, numberOfOriginalEqualityConstraints + numberOfActiveInequalityConstraints, 0);

      ATransposeMuAndCTransposeLambda.reshape(numberOfVariables, 1);
      CommonOps_DSCC.multTransA(AAndC, augmentedLagrangeMultipliers, ATransposeMuAndCTransposeLambda);

      CommonOps_DDRM.add(quadraticCostQVector, ATransposeMuAndCTransposeLambda, tempVector);

      CommonOps_DSCC.mult(QInverse, tempVector, xSolutionToPack);
      CommonOps_DDRM.scale(-1.0, xSolutionToPack);

      int startRow = 0;
      int numberOfRows = numberOfOriginalEqualityConstraints;
      CommonOps_DDRM.extract(augmentedLagrangeMultipliers, startRow, startRow + numberOfRows, 0, 1, lagrangeEqualityConstraintMultipliersToPack, 0, 0);

      startRow += numberOfRows;
      lagrangeInequalityConstraintMultipliersToPack.zero();
      for (int i = 0; i < numberOfActiveInequalityConstraints; i++)
      {
         int inequalityConstraintIndex = activeInequalityIndices.get(i);
         CommonOps_DDRM.extract(augmentedLagrangeMultipliers, startRow + i, startRow + i + 1, 0, 1, lagrangeInequalityConstraintMultipliersToPack, inequalityConstraintIndex, 0);
      }

      startRow += numberOfActiveInequalityConstraints;
      lagrangeLowerBoundConstraintMultipliersToPack.zero();
      for (int i = 0; i < numberOfActiveLowerBoundConstraints; i++)
      {
         int lowerBoundConstraintIndex = activeLowerBoundIndices.get(i);
         CommonOps_DDRM.extract(augmentedLagrangeMultipliers, startRow + i, startRow + i + 1, 0, 1, lagrangeLowerBoundConstraintMultipliersToPack, lowerBoundConstraintIndex, 0);
      }

      startRow += numberOfActiveLowerBoundConstraints;
      lagrangeUpperBoundConstraintMultipliersToPack.zero();
      for (int i = 0; i < numberOfActiveUpperBoundConstraints; i++)
      {
         int upperBoundConstraintIndex = activeUpperBoundIndices.get(i);

         CommonOps_DDRM.extract(augmentedLagrangeMultipliers, startRow + i, startRow + i + 1, 0, 1, lagrangeUpperBoundConstraintMultipliersToPack, upperBoundConstraintIndex, 0);
      }
   }

   public void getLagrangeEqualityConstraintMultipliers(DMatrixRMaj multipliersMatrixToPack)
   {
      multipliersMatrixToPack.set(lagrangeEqualityConstraintMultipliers);
   }

   public void getLagrangeInequalityConstraintMultipliers(DMatrixRMaj multipliersMatrixToPack)
   {
      multipliersMatrixToPack.set(lagrangeInequalityConstraintMultipliers);
   }

   public void getLagrangeLowerBoundsMultipliers(DMatrixRMaj multipliersMatrixToPack)
   {
      multipliersMatrixToPack.set(lagrangeLowerBoundMultipliers);
   }

   public void getLagrangeUpperBoundsMultipliers(DMatrixRMaj multipliersMatrixToPack)
   {
      multipliersMatrixToPack.set(lagrangeUpperBoundMultipliers);
   }

   public void setInverseCostCalculator(InverseCostCalculator<DMatrixSparseCSC> inverseSolver)
   {
      this.inverseSolver = inverseSolver;
   }

   private static class SparseInverseCostCalculator implements InverseCostCalculator<DMatrixSparseCSC>
   {
      private final LinearSolverSparse<DMatrixSparseCSC, DMatrixRMaj> inversionSolver = LinearSolverFactory_DSCC.lu(FillReducing.NONE);
      private final DMatrixSparseCSC identity = new DMatrixSparseCSC(0, 0);

      @Override
      public void computeInverse(DMatrixSparseCSC matrix, DMatrixSparseCSC inverseMatrix)
      {
         int numberOfVariables = matrix.getNumRows();

         identity.reshape(numberOfVariables, numberOfVariables);
         inverseMatrix.reshape(numberOfVariables, numberOfVariables);
         CommonOps_DSCC.setIdentity(identity);

         inversionSolver.setA(matrix);
         inversionSolver.solveSparse(identity, inverseMatrix);
      }
   }
}
