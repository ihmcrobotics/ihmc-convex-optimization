package us.ihmc.convexOptimization.linearProgram;

public enum SolverMethod
{
   /* Faster solve method. Is more prone to edge cases */
   SIMPLEX,

   /* Slower but more robust method */
   CRISS_CROSS,

   /* Performs linear feasibility test by solving only phase I Simplex */
   LINEAR_FEASIBILITY;
}
