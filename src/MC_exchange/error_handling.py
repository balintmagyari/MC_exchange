import sys
from mpi4py import MPI

def mpi_abort_on_exception(type, value, traceback):
    """Overrides default exception handling to ensure MPI aborts on all cores."""
    # Print the standard error message
    sys.__excepthook__(type, value, traceback)
    sys.stderr.flush()
    
    # Trigger the global kill switch
    print("Triggering MPI Abort across all ranks...")
    MPI.COMM_WORLD.Abort(1)