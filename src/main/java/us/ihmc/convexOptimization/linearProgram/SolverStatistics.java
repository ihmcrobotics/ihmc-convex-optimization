package us.ihmc.convexOptimization.linearProgram;

class SolverStatistics
{
   private double solveTime;
   private int iterations;
   private boolean foundSolution;

   public void clear()
   {
      solveTime = Double.NaN;
      iterations = 0;
      foundSolution = false;
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

   @Override
   public String toString()
   {
      return "Solve time: " + solveTime + "\nIterations: " + iterations + "\nFound solution: " + foundSolution + "\n";
   }
}
