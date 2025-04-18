# Parallel Multi-Objective Shortest Path (MOSP) Algorithm  
*A high-performance C/C++ implementation for dynamic graphs using MPI + OpenMP*

---

## 🎯 Description  
This project implements a **parallel algorithm** to compute Multi-Objective Shortest Paths (MOSP) in large dynamic networks (like road maps or social graphs). It uses:  
- **MPI** for distributed computing across multiple machines/nodes.  
- **OpenMP** for parallel processing within a single machine.  
- **METIS** to split graphs into smaller parts for efficient processing.  

**Use Cases**:  
- Transportation apps (find fastest + cheapest routes).  
- Network analysis (e.g., social media connections).  
- Scientific research on large graphs.  

