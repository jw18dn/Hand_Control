#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <mujoco/mujoco.h>

int main() {
    // Create a simple test case
    int nv = 3;  // 3 DOFs

    // Simulate MuJoCo output: jacp as [col0_x, col0_y, col0_z, col1_x, col1_y, col1_z, col2_x, col2_y, col2_z]
    std::vector<double> jacp = {
        1.0, 2.0, 3.0,   // column 0
        4.0, 5.0, 6.0,   // column 1
        7.0, 8.0, 9.0    // column 2
    };

    std::cout << "Raw MuJoCo data (column-major storage):\n";
    for (int i = 0; i < 9; i++) {
        std::cout << "jacp[" << i << "] = " << jacp[i] << "\n";
    }

    std::cout << "\n Expected 3x3 matrix:\n";
    std::cout << "  col0  col1  col2\n";
    std::cout << "[ 1.0   4.0   7.0 ]  row 0 (x)\n";
    std::cout << "[ 2.0   5.0   8.0 ]  row 1 (y)\n";
    std::cout << "[ 3.0   6.0   9.0 ]  row 2 (z)\n";

    // Test Eigen::Map with ColMajor (default)
    Eigen::Map<Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::ColMajor>> jacp_map(jacp.data(), 3, nv);

    std::cout << "\nEigen::Map with ColMajor:\n" << jacp_map << "\n";

    // Test with RowMajor
    Eigen::Map<Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor>> jacp_map_row(jacp.data(), 3, nv);

    std::cout << "\nEigen::Map with RowMajor:\n" << jacp_map_row << "\n";

    return 0;
}
