package us.ihmc.convexOptimization.linearProgram;

import gnu.trove.list.array.TIntArrayList;

public class SolverStatistics
{
   private double solveTime;
   private int iterations;
   private boolean foundSolution;

   /**
    * Given the inequality that parameterizes the LP: Ax <= b
    * Each index i in this list means that row i of A constrains the solution.
    */
   private final TIntArrayList activeSetIndices = new TIntArrayList();

   public void clear()
   {
      solveTime = Double.NaN;
      iterations = 0;
      foundSolution = false;
      activeSetIndices.reset();
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

   public void addActiveSetIndex(int index)
   {
      activeSetIndices.add(index);
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

   public TIntArrayList getActiveSetIndices()
   {
      return activeSetIndices;
   }

   @Override
   public String toString()
   {
      return "Solve time: " + solveTime + "\nIterations: " + iterations + "\nFound solution: " + foundSolution + "\n";
   }
}
