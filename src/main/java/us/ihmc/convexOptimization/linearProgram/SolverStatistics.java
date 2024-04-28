package us.ihmc.convexOptimization.linearProgram;

public class SolverStatistics
{
   /* Time taken by DictionaryFormLinearProgramSolver.solve{CrissCross, Simplex} in seconds */
   private double solveTime;

   /* Number of iterations (or pivots) taken by DictionaryFormLinearProgramSolver.solve{CrissCross, Simplex} */
   private int iterations;

   /* Whether DictionaryFormLinearProgramSolver.solve{CrissCross, Simplex} found a solution */
   private boolean foundSolution;

   /* Min entry of the d_rg for r in B-f. A zero entry means the solution is degenerate and additional pivots will not improve optimality */
   private double minDictionaryRHSColumnEntry;

   /* If foundSolution is false, reports failure reason */
   private LinearProgramFailureReason failureReason = null;

   /* If using Simplex and the solver failed, this is true if the failure occurred during Phase I */
   private boolean failedForPhaseI = false;

   public enum LinearProgramFailureReason
   {
      MAX_ITERATIONS_REACHED,
      NO_CANDIDATE_PIVOT,
      INVALID_PHASE_I_SOLUTION;
   }

   public void clear()
   {
      solveTime = Double.NaN;
      iterations = 0;
      foundSolution = false;
      minDictionaryRHSColumnEntry = Double.NaN;
      failureReason = null;
      failedForPhaseI = false;
   }

   public void setSolveTime(double solveTime)
   {
      this.solveTime = solveTime;
   }

   public int getAndIncrementIterations()
   {
      return iterations++;
   }

   public void setFoundSolution(boolean foundSolution)
   {
      this.foundSolution = foundSolution;
   }

   public void setMinDictionaryRHSColumnEntry(double minDictionaryRHSColumnEntry)
   {
      this.minDictionaryRHSColumnEntry = minDictionaryRHSColumnEntry;
   }

   public double getSolveTime()
   {
      return solveTime;
   }

   public int getIterations()
   {
      return iterations;
   }

   public boolean foundSolution()
   {
      return foundSolution;
   }

   public double getMinDictionaryRHSColumnEntry()
   {
      return minDictionaryRHSColumnEntry;
   }

   public void onSolutionFound()
   {
      foundSolution = true;
   }

   public void onSolverFailure(LinearProgramFailureReason failureReason, boolean failedForPhaseI)
   {
      foundSolution = false;
      this.failureReason = failureReason;
      this.failedForPhaseI = failedForPhaseI;
   }

   public LinearProgramFailureReason getFailureReason()
   {
      return failureReason;
   }

   public boolean isFailedForPhaseI()
   {
      return failedForPhaseI;
   }

   @Override
   public String toString()
   {
      return "Solve time: " + solveTime + "\nIterations: " + iterations + "\nFound solution: " + foundSolution + "\n";
   }
}
