package us.ihmc.convexOptimization.quadraticProgram;

import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;
import org.ejml.dense.row.factory.DecompositionFactory_DDRM;
import org.ejml.interfaces.decomposition.CholeskyDecomposition;

import gnu.trove.list.array.TIntArrayList;
import us.ihmc.commons.MathTools;
import us.ihmc.matrixlib.MatrixTools;
import us.ihmc.matrixlib.NativeCommonOps;

/**
 * Solves a Quadratic Program using an active set solver based on the Goldfarb-Idnani method. Should
 * work where some other simple active set solvers do not. This is the same algorithm in the
 * QuadProg++ and uQuadProg++ algorithms. Algorithm is fairly fast when it can find a solution. Uses
 * the algorithm found in the Paper "A numerically stable dual method for solving strictly convex
 * quadratic programs" by D. Goldfarb and A. Idnani.
 * <p>
 * The problem stored in the solver is of the form:</br>
 * min 0.5 * x G x + g0 x</br>
 * s.t.</br>
 * CE^T x + ce0 = 0</br>
 * CI^T x + ci0 >= 0</br>
 * </br>
 * To interface with the solver, however, use the standard form:</br>
 * min 0.5 * x G x + g0 x</br>
 * s.t.</br>
 * CE^T x = ce0</br>
 * CI^T x <= ci0
 * </p>
 *
 * @author Robert Griffin
 */
public class JavaQuadProgSolver extends AbstractSimpleActiveSetQPSolver
{
   private enum QuadProgStep
   {
      COMPUTE_CONSTRAINT_VIOLATIONS, FIND_MOST_VIOLATED_CONSTRAINT, COMPUTE_STEP_LENGTH, FAILED
   }

   private final static boolean bulkHandleEqualityConstraints = false;

   private static final int TRUE = 1;
   private static final int FALSE = 0;

   private boolean requireInequalityConstraintsSatisfied = true;

   private static final int defaultSize = 100;
   //private static final double epsilon = 1.0e-12;
   private static final double epsilon = 1.0e-24;

   private final DMatrixRMaj R = new DMatrixRMaj(0, 0);

   private final DMatrixRMaj inequalityConstraintViolations = new DMatrixRMaj(0, 0);
   private final DMatrixRMaj stepDirectionInPrimalSpace = new DMatrixRMaj(0, 0);
   private final DMatrixRMaj infeasibilityMultiplier = new DMatrixRMaj(0, 0);
   private final DMatrixRMaj d = new DMatrixRMaj(0, 0);

   private final DMatrixRMaj violatedConstraintNormal = new DMatrixRMaj(0, 0);
   private final DMatrixRMaj lagrangeMultipliers = new DMatrixRMaj(0, 0);
   private final DMatrixRMaj previousLagrangeMultipliers = new DMatrixRMaj(0, 0);
   private final DMatrixRMaj previousSolution = new DMatrixRMaj(0, 0);

   private final TIntArrayList activeSetIndices = new TIntArrayList(0);
   private final TIntArrayList previousActiveSetIndices = new TIntArrayList(0);
   private final TIntArrayList inactiveSetIndices = new TIntArrayList(0);
   private final TIntArrayList excludeConstraintFromActiveSet = new TIntArrayList(0); // booleans

   private final DMatrixRMaj J = new DMatrixRMaj(0, 0);
   private final DMatrixRMaj Q_augmented = new DMatrixRMaj(0, 0);
   private final DMatrixRMaj q_augmented = new DMatrixRMaj(0, 0);
   private final DMatrixRMaj Q_augmented_inv = new DMatrixRMaj(0, 0);

   private final CholeskyDecomposition<DMatrixRMaj> decomposer = DecompositionFactory_DDRM.chol(defaultSize, false);

   private final DMatrixRMaj decomposedQuadraticCostQMatrix = new DMatrixRMaj(0, 0);

   private final DMatrixRMaj totalLinearInequalityConstraintsCMatrix = new DMatrixRMaj(0, 0);
   private final DMatrixRMaj totalLinearInequalityConstraintsDVector = new DMatrixRMaj(0, 0);

   protected final DMatrixRMaj lowerBoundsCMatrix = new DMatrixRMaj(0, 0);
   protected final DMatrixRMaj upperBoundsCMatrix = new DMatrixRMaj(0, 0);

   private int problemSize;
   private int numberOfInequalityConstraints;
   private int totalNumberOfInequalityConstraints;
   private int numberOfEqualityConstraints;
   private int numberOfLowerBounds;
   private int numberOfUpperBounds;

   private int numberOfActiveConstraints;
   private double R_norm;
   private int constraintIndexForPartialStep;

   private int maxNumberOfIterations = 500;
   private double convergenceThreshold = 1.0e-14;
   //private double convergenceThreshold = Double.MIN_VALUE;

   protected final DMatrixRMaj computedObjectiveFunctionValue = new DMatrixRMaj(1, 1);

   public void setRequireInequalityConstraintsSatisfied(boolean requireInequalityConstraintsSatisfied)
   {
      this.requireInequalityConstraintsSatisfied = requireInequalityConstraintsSatisfied;
   }

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
      problemSize = 0;
      numberOfEqualityConstraints = 0;
      numberOfInequalityConstraints = 0;
      totalNumberOfInequalityConstraints = 0;
      numberOfLowerBounds = 0;
      numberOfUpperBounds = 0;

      quadraticCostQMatrix.reshape(0, 0);
      decomposedQuadraticCostQMatrix.reshape(0, 0);
      quadraticCostQVector.reshape(0, 0);

      linearEqualityConstraintsAMatrix.reshape(0, 0);
      linearEqualityConstraintsBVector.reshape(0, 0);

      linearInequalityConstraintsCMatrixO.reshape(0, 0);
      linearInequalityConstraintsDVectorO.reshape(0, 0);

      lowerBoundsCMatrix.reshape(0, 0);
      variableLowerBounds.reshape(0, 0);

      upperBoundsCMatrix.reshape(0, 0);
      variableUpperBounds.reshape(0, 0);
   }

   @Override
   public void setLowerBounds(DMatrixRMaj variableLowerBounds)
   {
      int numberOfLowerBounds = variableLowerBounds.getNumRows();
      if (numberOfLowerBounds != quadraticCostQMatrix.getNumRows())
         throw new RuntimeException("variableLowerBounds.getNumRows() != quadraticCostQMatrix.getNumRows()");

      lowerBoundsCMatrix.reshape(numberOfLowerBounds, numberOfLowerBounds);
      CommonOps_DDRM.setIdentity(lowerBoundsCMatrix);

      this.variableLowerBounds.set(variableLowerBounds);
      CommonOps_DDRM.scale(-1.0, this.variableLowerBounds);
   }

   @Override
   public void setUpperBounds(DMatrixRMaj variableUpperBounds)
   {
      int numberOfUpperBounds = variableUpperBounds.getNumRows();
      if (numberOfUpperBounds != quadraticCostQMatrix.getNumRows())
         throw new RuntimeException("variableUpperBounds.getNumRows() != quadraticCostQMatrix.getNumRows()");

      upperBoundsCMatrix.reshape(numberOfUpperBounds, numberOfUpperBounds);
      CommonOps_DDRM.setIdentity(upperBoundsCMatrix);
      CommonOps_DDRM.scale(-1.0, upperBoundsCMatrix);

      this.variableUpperBounds.set(variableUpperBounds);
   }

   @Override
   public void setQuadraticCostFunction(DMatrixRMaj costQuadraticMatrix, DMatrixRMaj costLinearVector, double quadraticCostScalar)
   {
      if (costLinearVector.getNumCols() != 1)
         throw new RuntimeException("costLinearVector.getNumCols() != 1");
      if (costQuadraticMatrix.getNumRows() != costLinearVector.getNumRows())
         throw new RuntimeException("costQuadraticMatrix.getNumRows() != costLinearVector.getNumRows()");
      if (costQuadraticMatrix.getNumRows() != costQuadraticMatrix.getNumCols())
         throw new RuntimeException("costQuadraticMatrix.getNumRows() != costQuadraticMatrix.getNumCols()");

      quadraticCostQMatrix.set(costQuadraticMatrix);
      quadraticCostQVector.set(costLinearVector);
      this.quadraticCostScalar = quadraticCostScalar;
   }

   @Override
   public double getObjectiveCost(DMatrixRMaj x)
   {
      NativeCommonOps.multQuad(x, quadraticCostQMatrix, computedObjectiveFunctionValue);
      CommonOps_DDRM.scale(0.5, computedObjectiveFunctionValue);
      CommonOps_DDRM.multAddTransA(quadraticCostQVector, x, computedObjectiveFunctionValue);
      return computedObjectiveFunctionValue.get(0, 0) + quadraticCostScalar;
   }

   @Override
   public void setLinearEqualityConstraints(DMatrixRMaj linearEqualityConstraintsAMatrix, DMatrixRMaj linearEqualityConstraintsBVector)
   {
      int numberOfEqualityConstraints = linearEqualityConstraintsBVector.getNumRows();

      if (linearEqualityConstraintsBVector.getNumCols() != 1)
         throw new RuntimeException("linearEqualityConstraintsBVector.getNumCols() != 1");
      if (linearEqualityConstraintsAMatrix.getNumRows() != linearEqualityConstraintsBVector.getNumRows())
         throw new RuntimeException("linearEqualityConstraintsAMatrix.getNumRows() != linearEqualityConstraintsBVector.getNumRows()");
      if (linearEqualityConstraintsAMatrix.getNumCols() != quadraticCostQMatrix.getNumCols())
         throw new RuntimeException("linearEqualityConstraintsAMatrix.getNumCols() != quadraticCostQMatrix.getNumCols()");

      this.linearEqualityConstraintsAMatrix.reshape(quadraticCostQMatrix.getNumCols(), numberOfEqualityConstraints);
      CommonOps_DDRM.transpose(linearEqualityConstraintsAMatrix, this.linearEqualityConstraintsAMatrix);
      CommonOps_DDRM.scale(-1.0, this.linearEqualityConstraintsAMatrix);

      this.linearEqualityConstraintsBVector.set(linearEqualityConstraintsBVector);
   }

   @Override
   public void setLinearInequalityConstraints(DMatrixRMaj linearInequalityConstraintCMatrix, DMatrixRMaj linearInequalityConstraintDVector)
   {
      int numberOfInequalityConstraints = linearInequalityConstraintDVector.getNumRows();

      if (linearInequalityConstraintDVector.getNumCols() != 1)
         throw new RuntimeException("linearInequalityConstraintDVector.getNumCols() != 1");
      if (linearInequalityConstraintCMatrix.getNumRows() != linearInequalityConstraintDVector.getNumRows())
         throw new RuntimeException("linearInequalityConstraintCMatrix.getNumRows() != linearInequalityConstraintDVector.getNumRows()");
      if (linearInequalityConstraintCMatrix.getNumCols() != quadraticCostQMatrix.getNumCols())
         throw new RuntimeException("linearInequalityConstraintCMatrix.getNumCols() != quadraticCostQMatrix.getNumCols()");

      linearInequalityConstraintsCMatrixO.reshape(quadraticCostQMatrix.getNumCols(), numberOfInequalityConstraints);
      CommonOps_DDRM.transpose(linearInequalityConstraintCMatrix, linearInequalityConstraintsCMatrixO);
      CommonOps_DDRM.scale(-1.0, linearInequalityConstraintsCMatrixO);

      linearInequalityConstraintsDVectorO.set(linearInequalityConstraintDVector);
   }

   @Override
   public void setUseWarmStart(boolean useWarmStart)
   {
      // TODO
   }

   @Override
   public void resetActiveSet()
   {
      // TODO
   }

   private final DMatrixRMaj lagrangeEqualityConstraintMultipliers = new DMatrixRMaj(0, 0);
   private final DMatrixRMaj lagrangeInequalityConstraintMultipliers = new DMatrixRMaj(0, 0);
   private final DMatrixRMaj lagrangeLowerBoundMultipliers = new DMatrixRMaj(0, 0);
   private final DMatrixRMaj lagrangeUpperBoundMultipliers = new DMatrixRMaj(0, 0);

   @Override
   public int solve(DMatrixRMaj solutionToPack)
   {
      numberOfEqualityConstraints = linearEqualityConstraintsBVector.getNumRows();
      numberOfLowerBounds = variableLowerBounds.getNumRows();
      numberOfUpperBounds = variableUpperBounds.getNumRows();
      numberOfInequalityConstraints = linearInequalityConstraintsDVectorO.getNumRows();
      problemSize = quadraticCostQMatrix.getNumCols();

      solutionToPack.reshape(problemSize, 1);
      solutionToPack.zero();
      lagrangeEqualityConstraintMultipliers.reshape(numberOfEqualityConstraints, 1);
      lagrangeEqualityConstraintMultipliers.zero();
      lagrangeInequalityConstraintMultipliers.reshape(numberOfInequalityConstraints, 1);
      lagrangeInequalityConstraintMultipliers.zero();
      lagrangeLowerBoundMultipliers.reshape(numberOfLowerBounds, 1);
      lagrangeLowerBoundMultipliers.zero();
      lagrangeUpperBoundMultipliers.reshape(numberOfUpperBounds, 1);
      lagrangeUpperBoundMultipliers.zero();

      reshape();
      zero();

      QuadProgStep currentStep = QuadProgStep.COMPUTE_CONSTRAINT_VIOLATIONS;

      double c1, c2;
      double stepLength; // step length, minimum of partial step (maximumStepInDualSpace) and full step (minimumStepInPrimalSpace);
      int mostViolatedConstraintIndex = 0; // this is the index of the constraint to be added to the active set

      J.reshape(problemSize, problemSize);

      /** Preprocessing phase */

      // compute the trace of the original matrix quadraticCostQMatrix
      c1 = CommonOps_DDRM.trace(quadraticCostQMatrix);

      // decompose the matrix quadraticCostQMatrix in the form L^T L
      decomposedQuadraticCostQMatrix.set(quadraticCostQMatrix);
      decomposer.decompose(decomposedQuadraticCostQMatrix);

      R_norm = 1.0; // this variable will hold the norm of the matrix R

      // compute the inverse of the factorized matrix G^-1, this is the initial value for H //// TODO: 5/14/17 combine this with the decomposition
      NativeCommonOps.invert(decomposedQuadraticCostQMatrix, J);
      c2 = CommonOps_DDRM.trace(J);

      int numberOfIterations = 0;

      if (bulkHandleEqualityConstraints)
      {
         if (numberOfEqualityConstraints > 0)
         {
            // TODO do some wild block operations in here to make things faster
            Q_augmented.reshape(problemSize + numberOfEqualityConstraints, problemSize + numberOfEqualityConstraints);
            q_augmented.reshape(problemSize + numberOfEqualityConstraints, 1);
            Q_augmented_inv.reshape(problemSize + numberOfEqualityConstraints, problemSize + numberOfEqualityConstraints);

            MatrixTools.setMatrixBlock(Q_augmented, 0, 0, quadraticCostQMatrix, 0, 0, problemSize, problemSize, 1.0);
            MatrixTools.setMatrixBlock(Q_augmented, 0, problemSize, linearEqualityConstraintsAMatrix, 0, 0, problemSize, numberOfEqualityConstraints, -1.0);

            tempMatrix.reshape(numberOfEqualityConstraints, problemSize);
            CommonOps_DDRM.transpose(linearEqualityConstraintsAMatrix, tempMatrix);

            MatrixTools.setMatrixBlock(Q_augmented, problemSize, 0, tempMatrix, 0, 0, numberOfEqualityConstraints, problemSize, 1.0);

            NativeCommonOps.invert(Q_augmented, Q_augmented_inv);

            MatrixTools.setMatrixBlock(q_augmented, 0, 0, quadraticCostQVector, 0, 0, problemSize, 1, -1.0);
            MatrixTools.setMatrixBlock(q_augmented, problemSize, 0, linearEqualityConstraintsBVector, 0, 0, numberOfEqualityConstraints, 1, -1.0);

            tempMatrix.reshape(problemSize + numberOfEqualityConstraints, 1);
            CommonOps_DDRM.mult(Q_augmented_inv, q_augmented, tempMatrix);
            MatrixTools.setMatrixBlock(solutionToPack, 0, 0, tempMatrix, 0, 0, problemSize, 1, 1.0);
            MatrixTools.setMatrixBlock(lagrangeMultipliers, 0, 0, tempMatrix, problemSize, 0, numberOfEqualityConstraints, 1, 1.0);

            // Add equality constraints to the working set A
            numberOfActiveConstraints = 0;
            for (int equalityConstraintIndex = 0; equalityConstraintIndex < numberOfEqualityConstraints; equalityConstraintIndex++)
            {
               MatrixTools.setMatrixBlock(violatedConstraintNormal, 0, 0, linearEqualityConstraintsAMatrix, 0, equalityConstraintIndex, problemSize, 1, 1.0);
               compute_d();

               activeSetIndices.set(equalityConstraintIndex, -equalityConstraintIndex - 1);

               if (!addConstraint())
               { // Constraints are linearly dependent
                  CommonOps_DDRM.fill(solutionToPack, Double.NaN);
                  CommonOps_DDRM.fill(lagrangeEqualityConstraintMultipliers, Double.POSITIVE_INFINITY);
                  CommonOps_DDRM.fill(lagrangeInequalityConstraintMultipliers, Double.POSITIVE_INFINITY);
                  CommonOps_DDRM.fill(lagrangeLowerBoundMultipliers, Double.POSITIVE_INFINITY);
                  CommonOps_DDRM.fill(lagrangeUpperBoundMultipliers, Double.POSITIVE_INFINITY);
                  return numberOfIterations - 1;
               }
            }
         }
         else
         {
            // c1 * c2 is an estimate for cond(G)

            // Find the unconstrained minimizer of the quadratic form 0.5 * x G x + g0 x
            // this is the feasible point in the dual space.
            // x = -G^-1 * g0 = -J * J^T * g0
            tempMatrix.reshape(problemSize, 1);
            CommonOps_DDRM.multTransA(J, quadraticCostQVector, tempMatrix);
            CommonOps_DDRM.mult(-1.0, J, tempMatrix, solutionToPack);

            // Add equality constraints to the working set A
            numberOfActiveConstraints = 0;
         }
      }
      else
      {
         // c1 * c2 is an estimate for cond(G)

         // Find the unconstrained minimizer of the quadratic form 0.5 * x G x + g0 x
         // this is the feasible point in the dual space.
         // x = -G^-1 * g0 = -J * J^T * g0
         tempMatrix.reshape(problemSize, 1);
         CommonOps_DDRM.multTransA(J, quadraticCostQVector, tempMatrix);
         CommonOps_DDRM.mult(-1.0, J, tempMatrix, solutionToPack);

         // Add equality constraints to the working set A
         numberOfActiveConstraints = 0;
         for (int equalityConstraintIndex = 0; equalityConstraintIndex < numberOfEqualityConstraints; equalityConstraintIndex++)
         {
            MatrixTools.setMatrixBlock(violatedConstraintNormal, 0, 0, linearEqualityConstraintsAMatrix, 0, equalityConstraintIndex, problemSize, 1, 1.0);

            compute_d();
            updateStepDirectionInPrimalSpace();
            updateInfeasibilityMultiplier();

            // compute full step length: i.e., the minimum step in primal space s.t. the constraint becomes feasible
            double stepLengthForEqualityConstraint = computeStepLengthForEqualityConstraint(solutionToPack, equalityConstraintIndex);

            // set x = x + minimumStepInPrimalSpace * stepDirectionInPrimalSpace
            CommonOps_DDRM.addEquals(solutionToPack, stepLengthForEqualityConstraint, stepDirectionInPrimalSpace);

            // set u = u+
            lagrangeMultipliers.set(numberOfActiveConstraints, stepLengthForEqualityConstraint);
            MatrixTools.addMatrixBlock(lagrangeMultipliers, 0, 0, infeasibilityMultiplier, 0, 0, numberOfActiveConstraints, 1, stepLengthForEqualityConstraint);

            // compute the new solution value
            activeSetIndices.set(equalityConstraintIndex, -equalityConstraintIndex - 1);

            if (!addConstraint())
            { // Constraints are linearly dependent
               CommonOps_DDRM.fill(solutionToPack, Double.NaN);
               CommonOps_DDRM.fill(lagrangeEqualityConstraintMultipliers, Double.POSITIVE_INFINITY);
               CommonOps_DDRM.fill(lagrangeInequalityConstraintMultipliers, Double.POSITIVE_INFINITY);
               CommonOps_DDRM.fill(lagrangeLowerBoundMultipliers, Double.POSITIVE_INFINITY);
               CommonOps_DDRM.fill(lagrangeUpperBoundMultipliers, Double.POSITIVE_INFINITY);
               return numberOfIterations - 1;
            }
         }
      }

      // set iai = K \ A
      for (int inequalityConstraintIndex = 0; inequalityConstraintIndex < totalNumberOfInequalityConstraints; inequalityConstraintIndex++)
         inactiveSetIndices.set(inequalityConstraintIndex, inequalityConstraintIndex);

      constraintIndexForPartialStep = 0;

      double fullStepLength;
      boolean isValid = true;

      while (true)
      {
         switch (currentStep)
         {
            case COMPUTE_CONSTRAINT_VIOLATIONS:

               if (computeConstraintViolations(solutionToPack, c1, c2))
               { // numerically there are not infeasibilities anymore
                 // the sum of all the constraint violations is negligible, so we are finished
                  partitionLagrangeMultipliers(lagrangeEqualityConstraintMultipliers,
                                               lagrangeInequalityConstraintMultipliers,
                                               lagrangeLowerBoundMultipliers,
                                               lagrangeUpperBoundMultipliers);
                  return numberOfIterations;
               }

            case FIND_MOST_VIOLATED_CONSTRAINT:
               // Step 2: check for feasibility and determine a new S-pair
               double biggestConstraintViolation = 0.0;
               mostViolatedConstraintIndex = 0;
               for (int inequalityConstraintIndex = 0; inequalityConstraintIndex < totalNumberOfInequalityConstraints; inequalityConstraintIndex++)
               { // select the constraint from the inactive set that is most violated
                  if (inequalityConstraintViolations.get(inequalityConstraintIndex) < biggestConstraintViolation
                        && inactiveSetIndices.get(inequalityConstraintIndex) != -1 && excludeConstraintFromActiveSet.get(inequalityConstraintIndex) == TRUE)
                  {
                     biggestConstraintViolation = inequalityConstraintViolations.get(inequalityConstraintIndex);
                     mostViolatedConstraintIndex = inequalityConstraintIndex;
                  }
               }

               if (biggestConstraintViolation >= 0.0)
               {
                  if (requireInequalityConstraintsSatisfied)
                  { // the active set clearly wasn't satisfied, so the solution isn't valid.
                     CommonOps_DDRM.fill(solutionToPack, Double.NaN);
                     CommonOps_DDRM.fill(lagrangeEqualityConstraintMultipliers, Double.POSITIVE_INFINITY);
                     CommonOps_DDRM.fill(lagrangeInequalityConstraintMultipliers, Double.POSITIVE_INFINITY);
                     CommonOps_DDRM.fill(lagrangeLowerBoundMultipliers, Double.POSITIVE_INFINITY);
                     CommonOps_DDRM.fill(lagrangeUpperBoundMultipliers, Double.POSITIVE_INFINITY);
                     return numberOfIterations;
                  }
                  else
                  { // we don't have any violations in the inactive set, so the current solution is valid, by assuming that the active set is satisfied
                     partitionLagrangeMultipliers(lagrangeEqualityConstraintMultipliers,
                                                  lagrangeInequalityConstraintMultipliers,
                                                  lagrangeLowerBoundMultipliers,
                                                  lagrangeUpperBoundMultipliers);
                     return numberOfIterations;
                  }
               }

               // set np = n(violatedConstraintIndex)
               MatrixTools.setMatrixBlock(violatedConstraintNormal,
                                          0,
                                          0,
                                          totalLinearInequalityConstraintsCMatrix,
                                          0,
                                          mostViolatedConstraintIndex,
                                          problemSize,
                                          1,
                                          1.0);
               // set u = [u 0]^T
               lagrangeMultipliers.set(numberOfActiveConstraints, 0.0);
               // add the violated constraint to the active set A
               activeSetIndices.set(numberOfActiveConstraints, mostViolatedConstraintIndex);

            case COMPUTE_STEP_LENGTH:
               // Step 2a: determine step direction
               compute_d();
               // compute z = H np: the step direction in the primal space (through J, see the paper)
               updateStepDirectionInPrimalSpace();
               // compute N* np (if activeSetSize > 0): the negative of the step direction in the dual space
               updateInfeasibilityMultiplier();

               // Step 2b: compute step length
               constraintIndexForPartialStep = 0;
               // Step 2b i:
               // Compute partial step length (maximum step in dual space without violating dual feasibility)
               double partialStepLength = computePartialStepLength();

               // Step 2b ii:
               // Compute full step length (minimum step in primal space such that the violated constraint becomes feasible
               fullStepLength = computeFullStepLength(mostViolatedConstraintIndex);

               // the step is chosen as the minimum of maximumStepInDualSpace and minimumStepInPrimalSpace
               stepLength = Math.min(partialStepLength, fullStepLength);

               break;
            case FAILED:
               CommonOps_DDRM.fill(solutionToPack, Double.NaN);
               CommonOps_DDRM.fill(lagrangeEqualityConstraintMultipliers, Double.POSITIVE_INFINITY);
               CommonOps_DDRM.fill(lagrangeInequalityConstraintMultipliers, Double.POSITIVE_INFINITY);
               CommonOps_DDRM.fill(lagrangeLowerBoundMultipliers, Double.POSITIVE_INFINITY);
               CommonOps_DDRM.fill(lagrangeUpperBoundMultipliers, Double.POSITIVE_INFINITY);
               return numberOfIterations;
            default:
               throw new RuntimeException("This is an empty state.");
         }

         if (!isValid)
            break;

         // Step 2c: determine new S-pair and take step:
         if (!Double.isFinite(stepLength))
         { // case (i): no step in primal or dual space, QP is infeasible
            CommonOps_DDRM.fill(solutionToPack, Double.NaN);
            CommonOps_DDRM.fill(lagrangeEqualityConstraintMultipliers, Double.POSITIVE_INFINITY);
            CommonOps_DDRM.fill(lagrangeInequalityConstraintMultipliers, Double.POSITIVE_INFINITY);
            CommonOps_DDRM.fill(lagrangeLowerBoundMultipliers, Double.POSITIVE_INFINITY);
            CommonOps_DDRM.fill(lagrangeUpperBoundMultipliers, Double.POSITIVE_INFINITY);
            return numberOfIterations;
         }
         else if (!Double.isFinite(fullStepLength))
         { // case (ii): step in dual space
            numberOfIterations++;

            if (numberOfIterations > maxNumberOfIterations)
               break;
            currentStep = takeStepInDualSpace(stepLength);
         }
         else
         { // case (iii): step in primal and dual space.
            numberOfIterations++;

            if (numberOfIterations > maxNumberOfIterations)
               break;
            currentStep = takeStepInPrimalAndDualSpace(solutionToPack, stepLength, fullStepLength, mostViolatedConstraintIndex);
         }
      }

      CommonOps_DDRM.fill(solutionToPack, Double.NaN);
      CommonOps_DDRM.fill(lagrangeEqualityConstraintMultipliers, Double.POSITIVE_INFINITY);
      CommonOps_DDRM.fill(lagrangeInequalityConstraintMultipliers, Double.POSITIVE_INFINITY);
      CommonOps_DDRM.fill(lagrangeLowerBoundMultipliers, Double.POSITIVE_INFINITY);
      CommonOps_DDRM.fill(lagrangeUpperBoundMultipliers, Double.POSITIVE_INFINITY);
      return numberOfIterations - 1;
   }

   private void compute_d()
   {
      // compute d = H^T * np
      CommonOps_DDRM.multTransA(J, violatedConstraintNormal, d);
   }

   // compute z = H np: the step direction in the primal space (through J, see the paper)
   private void updateStepDirectionInPrimalSpace()
   {
      // setting of z = J * d
      for (int i = 0; i < problemSize; i++)
      {
         double sum = 0.0;
         for (int j = numberOfActiveConstraints; j < problemSize; j++)
            sum += J.get(i, j) * d.get(j);

         stepDirectionInPrimalSpace.set(i, sum);
      }
   }

   // compute -N* np (if activeSetSize > 0): the step direction in the dual space
   private void updateInfeasibilityMultiplier()
   {
      // setting of r = -R^-1 * d
      for (int i = numberOfActiveConstraints - 1; i >= 0; i--)
      {
         double sum = 0.0;
         for (int j = i + 1; j < numberOfActiveConstraints; j++)
            sum += R.get(i, j) * infeasibilityMultiplier.get(j);

         infeasibilityMultiplier.set(i, (sum - d.get(i)) / R.get(i, i));
      }
   }

   private boolean computeConstraintViolations(DMatrixRMaj solutionToPack, double c1, double c2)
   {
      // step 1: choose a violated constraint
      for (int activeInequalityIndex = numberOfEqualityConstraints; activeInequalityIndex < numberOfActiveConstraints; activeInequalityIndex++)
      {
         int activeConstraintIndex = activeSetIndices.get(activeInequalityIndex);
         inactiveSetIndices.set(activeConstraintIndex, -1);
      }

      // compute s(x) = ci^T * x + ci0 for all elements of K \ A
      double totalInequalityViolation = 0.0; // this value will contain the sum of all infeasibilities
      for (int inequalityConstraintIndex = 0; inequalityConstraintIndex < totalNumberOfInequalityConstraints; inequalityConstraintIndex++)
      {
         excludeConstraintFromActiveSet.set(inequalityConstraintIndex, TRUE);
         double constraintValue = 0.0;
         for (int j = 0; j < problemSize; j++)
            constraintValue += totalLinearInequalityConstraintsCMatrix.get(j, inequalityConstraintIndex) * solutionToPack.get(j);
         constraintValue += totalLinearInequalityConstraintsDVector.get(inequalityConstraintIndex);
         inequalityConstraintViolations.set(inequalityConstraintIndex, constraintValue);
         totalInequalityViolation += Math.min(0.0, constraintValue);
      }

      if (Math.abs(totalInequalityViolation) < (1.0 + totalNumberOfInequalityConstraints) * convergenceThreshold * c1 * c2 * 100.0)
      { // numerically there are not infeasibilities anymore
         return true;
      }

      // save old values for u, x, and A
      MatrixTools.setMatrixBlock(previousLagrangeMultipliers, 0, 0, lagrangeMultipliers, 0, 0, numberOfActiveConstraints, 1, 1.0);
      previousSolution.set(solutionToPack);

      for (int i = 0; i < numberOfActiveConstraints; i++)
         previousActiveSetIndices.set(i, activeSetIndices.get(i));

      return false;
   }

   private double computeStepLengthForEqualityConstraint(DMatrixRMaj solutionToPack, int equalityConstraintIndex)
   {
      // compute full step length: i.e., the minimum step in primal space s.t. the constraint becomes feasible
      double fullStepLength = 0.0;
      if (!MathTools.epsilonEquals(CommonOps_DDRM.dot(stepDirectionInPrimalSpace, stepDirectionInPrimalSpace), 0.0, epsilon)) // i.e. z != 0
      {
         fullStepLength = (-CommonOps_DDRM.dot(violatedConstraintNormal, solutionToPack) - linearEqualityConstraintsBVector.get(equalityConstraintIndex))
               / CommonOps_DDRM.dot(stepDirectionInPrimalSpace, violatedConstraintNormal);
      }
      return fullStepLength;
   }

   private double computePartialStepLength()
   {
      // Compute partial step length (maximum step in dual space without violating dual feasibility)
      double partialStepLength = Double.POSITIVE_INFINITY;
      // find the constraintIndexForMinimumStepLength s.t. it reaches the minimum of u+(x) / r
      for (int k = numberOfEqualityConstraints; k < numberOfActiveConstraints; k++)
      {
         double minimumStepLength = -lagrangeMultipliers.get(k) / infeasibilityMultiplier.get(k);
         if (infeasibilityMultiplier.get(k) < 0.0 && minimumStepLength < partialStepLength)
         {
            partialStepLength = minimumStepLength;
            constraintIndexForPartialStep = activeSetIndices.get(k);
         }
      }

      return partialStepLength;
   }

   private double computeFullStepLength(int violatedConstraintIndex)
   {
      double fullStepLength = Double.POSITIVE_INFINITY;
      // Compute full step length (minimum step in primal space such that the violated constraint becomes feasible
      {
         fullStepLength = -inequalityConstraintViolations.get(violatedConstraintIndex) / CommonOps_DDRM.dot(stepDirectionInPrimalSpace, violatedConstraintNormal);
         if (fullStepLength < 0.0) // patch suggested by Takano Akio for handling numerical inconsistencies
            fullStepLength = Double.POSITIVE_INFINITY;
      }

      return fullStepLength;
   }

   private QuadProgStep takeStepInDualSpace(double stepLength)
   {
      // case (ii): step in dual space
      // set u = u + t * [r 1] and drop constraintIndexForMinimumStepLength from the active set
      MatrixTools.addMatrixBlock(lagrangeMultipliers, 0, 0, infeasibilityMultiplier, 0, 0, numberOfActiveConstraints, 1, stepLength);
      lagrangeMultipliers.set(numberOfActiveConstraints, lagrangeMultipliers.get(numberOfActiveConstraints) + stepLength);

      inactiveSetIndices.set(constraintIndexForPartialStep, constraintIndexForPartialStep);
      if (deleteConstraint(J))
         return QuadProgStep.COMPUTE_STEP_LENGTH;
      else
         return QuadProgStep.FAILED;
   }

   private QuadProgStep takeStepInPrimalAndDualSpace(DMatrixRMaj solutionToPack, double stepLength, double fullStepLength, int mostViolatedConstraintIndex)
   {
      CommonOps_DDRM.addEquals(solutionToPack, stepLength, stepDirectionInPrimalSpace);

      // u = u + t * [r 1]
      MatrixTools.addMatrixBlock(lagrangeMultipliers, 0, 0, infeasibilityMultiplier, 0, 0, numberOfActiveConstraints, 1, stepLength);
      lagrangeMultipliers.set(numberOfActiveConstraints, lagrangeMultipliers.get(numberOfActiveConstraints) + stepLength);

      if (MathTools.epsilonEquals(stepLength, fullStepLength, epsilon))
      { // full step has been taken, using the minimumStepInPrimalSpace
        // add the violated constraint to the active set
         if (!addConstraint())
         {
            if (!requireInequalityConstraintsSatisfied)
               excludeConstraintFromActiveSet.set(mostViolatedConstraintIndex, FALSE);

            if (!deleteConstraint(J))
               return QuadProgStep.FAILED;

            for (int i = 0; i < totalNumberOfInequalityConstraints; i++)
               inactiveSetIndices.set(i, i);

            for (int i = 0; i < numberOfActiveConstraints; i++)
            {
               activeSetIndices.set(i, previousActiveSetIndices.get(i));
               inactiveSetIndices.set(activeSetIndices.get(i), -1);
            }
            MatrixTools.setMatrixBlock(lagrangeMultipliers, 0, 0, previousLagrangeMultipliers, 0, 0, numberOfActiveConstraints, 1, 1.0);

            solutionToPack.set(previousSolution);

            return QuadProgStep.FIND_MOST_VIOLATED_CONSTRAINT;
         }
         else
         {
            inactiveSetIndices.set(mostViolatedConstraintIndex, -1);
         }

         return QuadProgStep.COMPUTE_CONSTRAINT_VIOLATIONS;
      }
      else
      { // a partial step has taken
        // drop constraint constraintIndexForMinimumStepLength
         inactiveSetIndices.set(constraintIndexForPartialStep, constraintIndexForPartialStep);
         if (!deleteConstraint(J))
            return QuadProgStep.FAILED;

         // update s[ip] = CI * x + ci0
         double sum = 0.0;
         for (int k = 0; k < problemSize; k++)
            sum += totalLinearInequalityConstraintsCMatrix.get(k, mostViolatedConstraintIndex) * solutionToPack.get(k);
         inequalityConstraintViolations.set(mostViolatedConstraintIndex, sum + totalLinearInequalityConstraintsDVector.get(mostViolatedConstraintIndex));

         return QuadProgStep.COMPUTE_STEP_LENGTH;
      }
   }

   private boolean addConstraint()
   {
      double cc, ss, h, t1, t2, xny;

      // we have to find the Givens rotation which will reduce the element d(j) to zero.
      // if it is already zero, we don't have to do anything, except of decreasing j
      for (int j = problemSize - 1; j >= numberOfActiveConstraints + 1; j--)
      {
         /*
          * The Givens rotation is done with the matrix (cc cs, cs -cc). If cc is one, then element (j) of d
          * is zero compared with element (j - 1). Hence we don't have to do anything. If cc is zero, then we
          * just have to switch column (j) and column (j - 1) of J. Since we only switch columns in J, we
          * have to be careful how we update d depending on the sign of gs. Otherwise we have to apply the
          * Givens rotation to these columns. The i - 1 element of d has to be updated to h.
          */
         cc = d.get(j - 1);
         ss = d.get(j);
         h = distance(cc, ss);
         if (MathTools.epsilonEquals(h, 0.0, epsilon)) // h == 0
            continue;
         d.set(j, 0.0);
         ss = ss / h;
         cc = cc / h;
         if (cc < 0.0)
         {
            cc = -cc;
            ss = -ss;
            d.set(j - 1, -h);
         }
         else
         {
            d.set(j - 1, h);
         }

         xny = ss / (1.0 + cc);
         for (int k = 0; k < problemSize; k++)
         {
            t1 = J.get(k, j - 1);
            t2 = J.get(k, j);
            J.set(k, j - 1, t1 * cc + t2 * ss);
            J.set(k, j, xny * (t1 + J.get(k, j - 1)) - t2);
         }
      }

      // update the number of constraints added
      numberOfActiveConstraints++;

      if (numberOfActiveConstraints > problemSize)
      {
         // problem is over constrained
         return false;
      }

      // To update R we have to put the numberOfActiveConstraints components of the d vector into column numberOfActiveConstraints - 1 of R
      for (int i = 0; i < numberOfActiveConstraints; i++)
         R.set(i, numberOfActiveConstraints - 1, d.get(i));

      if (Math.abs(d.get(numberOfActiveConstraints - 1)) < epsilon * R_norm)
      {
         // problem degenerate
         return false;
      }

      R_norm = Math.max(R_norm, Math.abs(d.get(numberOfActiveConstraints - 1)));
      return true;
   }

   /** the only time this should be called is when removing an inequality constraint **/
   private boolean deleteConstraint(DMatrixRMaj J)
   {
      double cc, ss, h, xny, t1, t2;
      int qq = -1;

      // Find the index qq for active constraintIndexForMinimumStepLength to be removed
      for (int i = numberOfEqualityConstraints; i < numberOfActiveConstraints; i++)
      {
         if (activeSetIndices.get(i) == constraintIndexForPartialStep)
         {
            qq = i;
            break;
         }
      }

      if (qq == -1)
         return false;

      // remove the constraint from the active set and the duals
      for (int i = qq; i < numberOfActiveConstraints - 1; i++)
      {
         activeSetIndices.set(i, activeSetIndices.get(i + 1));
         lagrangeMultipliers.set(i, lagrangeMultipliers.get(i + 1));

         for (int j = 0; j < problemSize; j++)
            R.set(j, i, R.get(j, i + 1));
      }

      // FIXME use the remove row feature
      activeSetIndices.set(numberOfActiveConstraints - 1, activeSetIndices.get(numberOfActiveConstraints));
      lagrangeMultipliers.set(numberOfActiveConstraints - 1, lagrangeMultipliers.get(numberOfActiveConstraints));
      activeSetIndices.set(numberOfActiveConstraints, 0);
      lagrangeMultipliers.set(numberOfActiveConstraints, 0.0);
      for (int j = 0; j < numberOfActiveConstraints; j++)
         R.set(j, numberOfActiveConstraints - 1, 0.0);

      // constraint has been fully removed
      numberOfActiveConstraints--;

      if (numberOfActiveConstraints == 0)
         return true;

      for (int j = qq + numberOfEqualityConstraints; j < numberOfActiveConstraints; j++)
      {
         cc = R.get(j, j);
         ss = R.get(j + 1, j);
         h = distance(cc, ss);

         if (MathTools.epsilonEquals(h, 0.0, epsilon)) // h == 0
            continue;

         cc = cc / h;
         ss = ss / h;
         R.set(j + 1, j, 0.0);

         if (cc < 0.0)
         {
            R.set(j, j, -h);
            cc = -cc;
            ss = -ss;
         }
         else
         {
            R.set(j, j, h);
         }

         xny = ss / (1.0 + cc);
         for (int k = j + 1; k < numberOfActiveConstraints; k++)
         {
            t1 = R.get(j, k);
            t2 = R.get(j + 1, k);
            R.set(j, k, t1 * cc + t2 * ss);
            R.set(j + 1, k, xny * (t1 + R.get(j, k)) - t2);
         }

         for (int k = 0; k < problemSize; k++)
         {
            t1 = J.get(k, j);
            t2 = J.get(k, j + 1);
            J.set(k, j, t1 * cc + t2 * ss);
            J.set(k, j + 1, xny * (J.get(k, j) + t1) - t2);
         }
      }

      return true;
   }

   public void reshape()
   {
      int numberOfInequalityConstraints = linearInequalityConstraintsDVectorO.getNumRows();
      int numberOfLowerBounds = variableLowerBounds.getNumRows();
      int numberOfUpperBounds = variableUpperBounds.getNumRows();
      int numberOfConstraints = numberOfEqualityConstraints + numberOfInequalityConstraints + numberOfLowerBounds + numberOfUpperBounds;

      totalNumberOfInequalityConstraints = numberOfInequalityConstraints + numberOfLowerBounds + numberOfUpperBounds;

      R.reshape(problemSize, problemSize);
      inequalityConstraintViolations.reshape(totalNumberOfInequalityConstraints, 1);
      stepDirectionInPrimalSpace.reshape(problemSize, 1);
      infeasibilityMultiplier.reshape(numberOfConstraints, 1);
      d.reshape(problemSize, 1);
      violatedConstraintNormal.reshape(problemSize, 1);
      lagrangeMultipliers.reshape(numberOfConstraints, 1);
      previousSolution.reshape(problemSize, 1);
      previousLagrangeMultipliers.reshape(numberOfConstraints, 1);

      activeSetIndices.resetQuick();
      previousActiveSetIndices.resetQuick();
      inactiveSetIndices.resetQuick();
      excludeConstraintFromActiveSet.resetQuick();

      activeSetIndices.fill(0, numberOfConstraints, 0);
      previousActiveSetIndices.fill(0, numberOfConstraints, 0);
      inactiveSetIndices.fill(0, totalNumberOfInequalityConstraints, 0);
      excludeConstraintFromActiveSet.fill(0, totalNumberOfInequalityConstraints, FALSE);

      // compile all the inequality constraints into one matrix
      totalLinearInequalityConstraintsCMatrix.reshape(problemSize, numberOfInequalityConstraints + numberOfLowerBounds + numberOfUpperBounds);
      totalLinearInequalityConstraintsDVector.reshape(numberOfInequalityConstraints + numberOfLowerBounds + numberOfUpperBounds, 1);

      // add inequality constraints to total inequality constraint
      MatrixTools.setMatrixBlock(totalLinearInequalityConstraintsCMatrix,
                                 0,
                                 0,
                                 linearInequalityConstraintsCMatrixO,
                                 0,
                                 0,
                                 problemSize,
                                 numberOfInequalityConstraints,
                                 1.0);
      MatrixTools.setMatrixBlock(totalLinearInequalityConstraintsDVector,
                                 0,
                                 0,
                                 linearInequalityConstraintsDVectorO,
                                 0,
                                 0,
                                 numberOfInequalityConstraints,
                                 1,
                                 1.0);

      // add lower bounds to total inequality constraint
      MatrixTools.setMatrixBlock(totalLinearInequalityConstraintsCMatrix,
                                 0,
                                 numberOfInequalityConstraints,
                                 lowerBoundsCMatrix,
                                 0,
                                 0,
                                 problemSize,
                                 numberOfLowerBounds,
                                 1.0);
      MatrixTools.setMatrixBlock(totalLinearInequalityConstraintsDVector,
                                 numberOfInequalityConstraints,
                                 0,
                                 variableLowerBounds,
                                 0,
                                 0,
                                 numberOfLowerBounds,
                                 1,
                                 1.0);

      // add upper bounds to total inequality constraint
      MatrixTools.setMatrixBlock(totalLinearInequalityConstraintsCMatrix,
                                 0,
                                 numberOfInequalityConstraints + numberOfLowerBounds,
                                 upperBoundsCMatrix,
                                 0,
                                 0,
                                 problemSize,
                                 numberOfUpperBounds,
                                 1.0);
      MatrixTools.setMatrixBlock(totalLinearInequalityConstraintsDVector,
                                 numberOfInequalityConstraints + numberOfLowerBounds,
                                 0,
                                 variableUpperBounds,
                                 0,
                                 0,
                                 numberOfUpperBounds,
                                 1,
                                 1.0);
   }

   // TODO make this more efficient
   private void partitionLagrangeMultipliers(DMatrixRMaj lagrangeEqualityConstraintMultipliersToPack,
                                             DMatrixRMaj lagrangeInequalityConstraintMultipliersToPack, DMatrixRMaj lagrangeLowerBoundMultipliersToPack,
                                             DMatrixRMaj lagrangeUpperBoundMultipliersToPack)
   {
      MatrixTools.setMatrixBlock(lagrangeEqualityConstraintMultipliersToPack, 0, 0, lagrangeMultipliers, 0, 0, numberOfEqualityConstraints, 1, 1.0);

      for (int inequalityConstraintNumber = numberOfEqualityConstraints; inequalityConstraintNumber < numberOfActiveConstraints; inequalityConstraintNumber++)
      {
         int inequalityConstraintIndex = activeSetIndices.get(inequalityConstraintNumber);

         if (inequalityConstraintIndex < 0)
            continue;

         if (inequalityConstraintIndex < numberOfInequalityConstraints)
         { // add to the inequality constraint lagrange multipliers
            lagrangeInequalityConstraintMultipliersToPack.set(inequalityConstraintIndex, 0, lagrangeMultipliers.get(inequalityConstraintNumber));
         }
         else if (inequalityConstraintIndex < numberOfInequalityConstraints + numberOfLowerBounds)
         { // add to the lower bound lagrange multipliers
            int localIndex = inequalityConstraintIndex - numberOfInequalityConstraints;
            lagrangeLowerBoundMultipliersToPack.set(localIndex, 0, lagrangeMultipliers.get(inequalityConstraintNumber));
         }
         else
         { // add to the upper bound lagrange multipliers
            int localIndex = inequalityConstraintIndex - numberOfInequalityConstraints - numberOfLowerBounds;
            lagrangeUpperBoundMultipliersToPack.set(localIndex, 0, lagrangeMultipliers.get(inequalityConstraintNumber));
         }
      }
   }

   private void zero()
   {
      R.zero();
      inequalityConstraintViolations.zero();
      stepDirectionInPrimalSpace.zero();
      infeasibilityMultiplier.zero();
      d.zero();
      violatedConstraintNormal.zero();
      lagrangeMultipliers.zero();
      previousSolution.zero();
      previousLagrangeMultipliers.zero();
   }

   /**
    * Computes the Euclidean distance between two numbers
    * 
    * @param a first number
    * @param b second number
    * @return Euclidean distance
    */
   private static double distance(double a, double b)
   {
      double a1 = Math.abs(a);
      double b1 = Math.abs(b);
      if (a1 > b1)
      {
         double t = b1 / a1;
         return a1 * Math.sqrt(1.0 + t * t);
      }
      else if (b1 > a1)
      {
         double t = a1 / b1;
         return b1 * Math.sqrt(1.0 + t * t);
      }

      return a1 * Math.sqrt(2.0);
   }

   private final DMatrixRMaj tempMatrix = new DMatrixRMaj(defaultSize, defaultSize);

   @Override
   public void getLagrangeEqualityConstraintMultipliers(DMatrixRMaj multipliersMatrixToPack)
   {
      multipliersMatrixToPack.set(lagrangeEqualityConstraintMultipliers);
   }

   @Override
   public void getLagrangeInequalityConstraintMultipliers(DMatrixRMaj multipliersMatrixToPack)
   {
      multipliersMatrixToPack.set(lagrangeInequalityConstraintMultipliers);
   }

   @Override
   public void getLagrangeLowerBoundsMultipliers(DMatrixRMaj multipliersMatrixToPack)
   {
      multipliersMatrixToPack.set(lagrangeLowerBoundMultipliers);
   }

   @Override
   public void getLagrangeUpperBoundsMultipliers(DMatrixRMaj multipliersMatrixToPack)
   {
      multipliersMatrixToPack.set(lagrangeUpperBoundMultipliers);
   }
}
