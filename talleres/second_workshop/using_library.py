from linearAlgebra import Vector, Matrix
from linearAlgebra import dot_product, magnitude, normalize, cross_product, angle_between
from linearAlgebra import scale, add, subtract, vector_multiply, matrix_multiply
from linearAlgebra import transpose, determinant, inverse, identity_matrix, zeros_matrix, ones_matrix


def vector_examples():
    """Examples of Vector class usage."""
    print("=== VECTOR EXAMPLES ===")

    # Create vectors
    v1 = Vector([1, 2, 3])
    v2 = Vector([4, 5, 6])

    print(f"Vector v1: {v1}")
    print(f"Vector v2: {v2}")

    # Basic operations
    print(f"v1 + v2 = {v1 + v2}")
    print(f"v1 - v2 = {v1 - v2}")
    print(f"v1 * 2 = {v1 * 2}")
    print(f"v1 / 2 = {v1 / 2}")

    # Properties
    print(f"Magnitude of v1: {v1.magnitude}")
    print(f"Unit vector of v1: {v1.unit_vector}")

    # Products
    print(f"Dot product v1 · v2: {v1.dot(v2)}")
    print(f"Cross product v1 × v2: {v1.cross(v2)}")
    print(f"Angle between v1 and v2: {v1.angle_with(v2)} radians")

    # Using module functions
    print(f"Dot product (function): {dot_product(v1, v2)}")
    print(f"Magnitude (function): {magnitude(v1)}")
    print(f"Normalized vector (function): {normalize(v1)}")


def matrix_examples():
    """Examples of Matrix class usage."""
    print("\n=== MATRIX EXAMPLES ===")

    # Create matrices
    m1 = Matrix([[1, 2], [3, 4]])
    m2 = Matrix([[5, 6], [7, 8]])
    v = Vector([1, 2])

    print(f"Matrix m1:\n{m1}")
    print(f"Matrix m2:\n{m2}")
    print(f"Vector v: {v}")

    # Basic properties
    print(f"Shape of m1: {m1.shape}")
    print(f"Number of rows: {m1.num_rows}")
    print(f"Number of columns: {m1.num_columns}")

    # Basic operations
    print(f"m1 + m2:\n{m1 + m2}")
    print(f"m1 - m2:\n{m1 - m2}")
    print(f"m1 * 2:\n{m1 * 2}")

    # Advanced properties
    print(f"Transpose of m1:\n{m1.T}")
    print(f"Trace of m1: {m1.trace}")
    print(f"Determinant of m1: {m1.determinant}")
    print(f"Is square?: {m1.is_square()}")

    # Multiplications
    print(f"m1 * v:\n{m1 * v}")
    print(f"m1 * m2:\n{m1 * m2}")

    # Using module functions
    print(f"Addition (function):\n{add(m1, m2)}")
    print(
        f"Matrix-vector multiplication (function):\n{vector_multiply(m1, v)}")
    print(
        f"Matrix-matrix multiplication (function):\n{matrix_multiply(m1, m2)}")


def special_matrices_examples():
    """Examples of special matrix creation."""
    print("\n=== SPECIAL MATRICES ===")

    # Identity matrix
    I = identity_matrix(3)
    print(f"3x3 Identity matrix:\n{I}")

    # Zero matrix
    zeros = zeros_matrix(2, 3)
    print(f"2x3 Zero matrix:\n{zeros}")

    # Ones matrix
    ones = ones_matrix(3, 2)
    print(f"3x2 Ones matrix:\n{ones}")


if __name__ == "__main__":
    print("Linear Algebra Library Usage Examples")
    print("=" * 50)

    try:
        vector_examples()
        matrix_examples()
        special_matrices_examples()
    except NotImplementedError:
        print("\nFunctions are not implemented yet!")
        print("Students must complete the implementations in linAlg.py")
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure the functions are implemented correctly.")
