import math
from typing import Union, Tuple, List


class Vector:
    """
    Class for representing and manipulating vectors.

    A vector is a list of numbers that can represent
    points in space, directions, or any ordered sequence of values.
    """

    def __init__(self, components: List[Union[int, float]]):
        """
        Initializes a vector with its components.

        Args:
            components: List of numbers representing the components of the vector
        """
        self.components = components

    def __str__(self) -> str:
        """String representation of the Vector"""
        return ", ".join(str(c) for c in self.components)

    def __repr__(self) -> str:
        """Detailed representation of the Vector"""
        return f"Vector({self.components})"

    def __len__(self) -> int:
        """Return the vector's len"""
        return len(self.components)

    def __getitem__(self, index: int) -> Union[int, float]:
        """Allows access to vector components using indices."""
        return self.components[index]

    def __setitem__(self, index: int, value: Union[int, float]):
        """Allows modification of vector components using indices."""
        self.components[index] = value

    def __add__(self, other: 'Vector') -> 'Vector':
        """Adds two vectors using the + operator."""
        if len(self.components) != len(other.components):
            raise ValueError("Vectors must be of the same dimension to add.")
        result = [self[i] + other[i] for i in range(len(self.components))]
        return Vector(result)

    def __sub__(self, other: 'Vector') -> 'Vector':
        """Subtracts two vectors using the - operator."""
        if len(self.components) != len(other.components):
            raise ValueError(
                "Vectors must be of the same dimension to subtract.")
        result = [self[i] - other[i] for i in range(len(self.components))]
        return Vector(result)

    def __mul__(self, scalar: Union[int, float]) -> 'Vector':
        """Scalar multiplication using the * operator."""
        return Vector([self[i] * scalar for i in range(len(self))])

    def __rmul__(self, scalar: Union[int, float]) -> 'Vector':
        """Scalar multiplication (reversed order)."""
        return self * scalar

    def __ne__(self, other: 'Vector') -> bool:
        """Checks inequality between two vectors using the != operator."""
        return not self == other

    def __eq__(self, other: 'Vector') -> bool:
        """Checks equality between two vectors using the == operator."""
        return self.components == other.components

    def __truediv__(self, scalar: Union[int, float]) -> 'Vector':
        """Scalar division using the / operator."""
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide by zero.")
        return Vector([self[i] / scalar for i in range(len(self))])

    @property
    def magnitude(self) -> float:
        """Calculates and returns the magnitude (norm) of the vector."""
        return sum(c**2 for c in self.components) ** 0.5

    @property
    def unit_vector(self) -> 'Vector':
        """Returns the unit (normalized) vector."""
        mag = self.magnitude
        if mag == 0:
            raise ValueError("Cannot normalize a vector with zero magnitude.")
        normalized_components = [c / mag for c in self.components]
        return Vector(normalized_components)

    def dot(self, other: 'Vector') -> float:
        """
        Calculates the dot product with another vector.

        Args:
            other: Another vector to compute the dot product with.

        Returns:
            The dot product as a float.
        """
        if len(self.components) != len(other.components):
            raise ValueError(
                "Vectors must be of the same dimension to compute dot product.")

        return sum(a * b for a, b in zip(self.components, other.components))

    def cross(self, other: 'Vector') -> 'Vector':
        """
        Computes the cross product with another vector (only for 3D vectors).

        Args:
            other: Another vector to compute the cross product with.

        Returns:
            A new Vector resulting from the cross product.
        """
        if len(self.components) != 3 or len(other.components) != 3:
            raise ValueError("Cross product is only defined for 3D vectors.")

        a1, a2, a3 = self.components
        b1, b2, b3 = other.components

        cross_components = [
            a2 * b3 - a3 * b2,
            a3 * b1 - a1 * b3,
            a1 * b2 - a2 * b1
        ]

        return Vector(cross_components)

    def angle_with(self, other: 'Vector') -> float:
        """
        Calculates the angle between this vector and another.

        Args:
            other: Another vector.

        Returns:
            The angle in radians.
        """
        dot_prod = self.dot(other)
        mag_self = self.magnitude
        mag_other = other.magnitude

        if mag_self == 0 or mag_other == 0:
            raise ValueError(
                "Cannot compute angle with zero-magnitude vector.")

        return math.acos(dot_prod / (mag_self * mag_other))


class Matrix:
    """
    Clase para representar y manipular matrices.

    Una matriz es una colección rectangular de números organizados en filas y columnas.
    """

    def __init__(self, data: List[List[Union[int, float]]]):
        """
        Initializes a matrix with its data.

        Args:
            data: A list of lists representing the rows of the matrix.
        """
        if not data or not all(isinstance(row, list) for row in data):
            raise ValueError("Data must be a non-empty list of lists.")
        if not all(len(row) == len(data[0]) for row in data):
            raise ValueError("All rows must have the same number of columns.")

        self.entries = data
        self.rows = len(data)
        self.cols = len(data[0])

    def __len__(self) -> int:
        """Return the vector's len"""
        return self.rows

    def __str__(self) -> str:
        """String representation of the matrix"""
        return '\n'.join(' '.join(f'{entry}' for entry in row) for row in self.entries)

    def __repr__(self) -> str:
        """Detailed representation of the matrix."""
        return f"Matrix({self.entries})"

    def __getitem__(self, key: Union[int, Tuple[int, int]]) -> Union[List[Union[int, float]], Union[int, float]]:
        """Allows access to specific rows or elements of the matrix."""
        if isinstance(key, int):
            return self.entries[key]
        elif isinstance(key, tuple) and len(key) == 2:
            row, col = key
            return self.entries[row][col]
        else:
            raise ValueError(
                "Key must be an integer (for row) or a tuple of two integers (for element).")

    def __setitem__(self, key: Union[int, Tuple[int, int]], value: Union[List[Union[int, float]], int, float]):
        """
        Allows modification of entire rows or specific elements in the matrix.

        - If `key` is an integer, it replaces the entire row with a list of values.
        - If `key` is a tuple (row, column), it updates a single element.
        """
        if isinstance(key, int):
            if isinstance(value, list) and all(isinstance(v, (int, float)) for v in value):
                self.entries[key] = value
            else:
                raise ValueError(
                    "To modify a row, the value must be a list of integers or floats.")
        elif isinstance(key, tuple) and len(key) == 2:
            row, column = key
            if isinstance(row, int) and isinstance(column, int) and isinstance(value, (int, float)):
                self.entries[row][column] = value
            else:
                raise ValueError(
                    "To modify an element, the key must be a tuple of two integers and the value must be an integer or float.")
        else:
            raise ValueError(
                "The key must be either an integer or a tuple of two integers.")

    def __add__(self, other: 'Matrix') -> 'Matrix':
        """Adds two matrices using the + operator."""

        if other.rows != self.rows or other.cols != self.cols:
            raise ValueError(
                "To add matrices, they must have the same dimensions.")

        result_matrix = []
        for row in range(self.rows):
            row_aux = []
            for column in range(self.cols):
                row_aux += [self.entries[row][column] +
                            other.entries[row][column]]
            result_matrix.append(row_aux)

        return Matrix(result_matrix)

    def __sub__(self, other: 'Matrix') -> 'Matrix':
        """Subtract two matrices using the - operator."""

        if other.rows != self.rows or other.cols != self.cols:
            raise ValueError(
                "To Subtract matrices, they must have the same dimensions.")

        result_matrix = []
        for row in range(self.rows):
            row_aux = []
            for column in range(self.cols):
                row_aux += [self.entries[row][column] -
                            other.entries[row][column]]
            result_matrix.append(row_aux)

        return Matrix(result_matrix)

    def __mul__(self, other: Union['Matrix', 'Vector', int, float]) -> Union['Matrix', 'Vector']:
        """Multiplicación de matrices/vectores/escalares usando el operador *."""

        # Scalar multiplication (most efficient case)
        if isinstance(other, (int, float)):
            result_matrix = [[element * other for element in row]
                             for row in self.entries]
            return Matrix(result_matrix)

        # Matrix/Vector multiplication
        if isinstance(other, (Matrix, Vector)):
            other_rows = len(other)

            # Dimension check
            if self.cols != other_rows:
                raise ValueError(
                    "To multiply matrices of dimensions nxm and mxd, the number of columns of the first must equal the number of rows of the second")

            # Determine result dimensions
            is_matrix = isinstance(other, Matrix)
            other_cols = other.cols if is_matrix else 1

            # Pre-allocate result matrix for better performance
            result = [[0] * other_cols for _ in range(self.rows)]

            # Matrix multiplication using proper indexing
            for i in range(self.rows):
                for j in range(other_cols):
                    accumulator = 0
                    for k in range(self.cols):
                        if is_matrix:
                            accumulator += self.entries[i][k] * other[k][j]
                        else:
                            accumulator += self.entries[i][k] * other[k]
                    result[i][j] = accumulator

            # Return appropriate type
            if is_matrix:
                return Matrix(result)
            else:
                # For vector result, extract the single column
                return Vector([row[0] for row in result])

        else:
            raise ValueError("Operand must be a Matrix, Vector, int, or float")

    def __rmul__(self, scalar: Union[int, float]) -> 'Matrix':
        """Scalar multiplication (reversed order)."""
        return self * scalar  # Delegate to __mul__ since scalar multiplication is commutative

    def __eq__(self, other: 'Matrix') -> bool:
        """Equality between matrices using the == operator."""
        if not isinstance(other, Matrix):
            return False

        # Check dimensions first
        if self.rows != other.rows or self.cols != other.cols:
            return False

        # Compare all entries
        for i in range(self.rows):
            for j in range(self.cols):
                if self.entries[i][j] != other.entries[i][j]:
                    return False
        return True

    def __ne__(self, other: 'Matrix') -> bool:
        """Desigualdad entre matrices usando el operador !=."""
        return not self.__eq__(other)

    @property
    def num_rows(self) -> int:
        """Retorna el número de filas de la matriz."""
        return self.rows

    @property
    def num_columns(self) -> int:
        """Retorna el número de columnas de la matriz."""
        return self.cols

    @property
    def shape(self) -> Tuple[int, int]:
        """Return the matrix dimensions."""
        return (self.rows, self.cols)

    @property
    def T(self) -> 'Matrix':
        """Returns the transposed matrix"""
        transposed = [[self.entries[j][i]
                       for j in range(self.rows)] for i in range(self.cols)]
        return Matrix(transposed)

    @property
    def trace(self) -> Union[int, float]:
        """Calcula y retorna la traza de la matriz (suma de elementos diagonales)."""
        if not self.is_square():
            raise ValueError("Trace is only defined for square matrices")

        return sum(self.entries[i][i] for i in range(self.rows))

    @property
    def determinant(self) -> Union[int, float]:
        """Calculate and returns the Matrix determinant"""
        if not self.is_square():
            raise ValueError("Determinant is only defined for square matrices")

        n = self.rows

        # Base cases
        if n == 1:
            return self.entries[0][0]
        elif n == 2:
            return self.entries[0][0] * self.entries[1][1] - self.entries[0][1] * self.entries[1][0]

        # For larger matrices, use LU decomposition method
        # Create a copy to avoid modifying original matrix
        matrix = [row[:] for row in self.entries]

        # Convert to upper triangular using Gaussian elimination
        det = 1
        for i in range(n):
            # Find pivot
            max_row = i
            for k in range(i + 1, n):
                if abs(matrix[k][i]) > abs(matrix[max_row][i]):
                    max_row = k

            # Swap rows if needed
            if max_row != i:
                matrix[i], matrix[max_row] = matrix[max_row], matrix[i]
                det *= -1  # Row swap changes sign

            # Check for zero pivot
            if abs(matrix[i][i]) < 1e-10:
                return 0

            det *= matrix[i][i]

            # Eliminate below pivot
            for k in range(i + 1, n):
                factor = matrix[k][i] / matrix[i][i]
                for j in range(i, n):
                    matrix[k][j] -= factor * matrix[i][j]

        return det

    @property
    def inverse(self) -> 'Matrix':
        """Calculate and returns the inverse of the matrix"""
        if not self.is_square():
            raise ValueError("Inverse is only defined for square matrices")

        det = self.determinant
        if abs(det) < 1e-10:
            raise ValueError("Matrix is singular (determinant is zero)")

        n = self.rows

        # Special case for 2x2 matrix
        if n == 2:
            a, b = self.entries[0][0], self.entries[0][1]
            c, d = self.entries[1][0], self.entries[1][1]
            inv_entries = [[d/det, -b/det], [-c/det, a/det]]
            return Matrix(inv_entries)

        # For larger matrices, use Gauss-Jordan elimination
        # Create augmented matrix [A|I]
        augmented = []
        for i in range(n):
            row = self.entries[i][:] + [0] * n
            row[n + i] = 1  # Identity matrix part
            augmented.append(row)

        # Gauss-Jordan elimination
        for i in range(n):
            # Find pivot
            max_row = i
            for k in range(i + 1, n):
                if abs(augmented[k][i]) > abs(augmented[max_row][i]):
                    max_row = k
            augmented[i], augmented[max_row] = augmented[max_row], augmented[i]

            # Scale pivot row
            pivot = augmented[i][i]
            for j in range(2 * n):
                augmented[i][j] /= pivot

            # Eliminate column
            for k in range(n):
                if k != i:
                    factor = augmented[k][i]
                    for j in range(2 * n):
                        augmented[k][j] -= factor * augmented[i][j]

        # Extract inverse matrix
        inverse_entries = [[augmented[i][j + n]
                            for j in range(n)] for i in range(n)]
        return Matrix(inverse_entries)

    def is_square(self) -> bool:
        """Verifies if it is a square matrix"""
        return self.rows == self.cols

    def is_symmetric(self) -> bool:
        """Verifies if it is a symmetric matrix"""
        if not self.is_square():
            return False

        for i in range(self.rows):
            for j in range(self.cols):
                if self.entries[i][j] != self.entries[j][i]:
                    return False
        return True

    def is_diagonal(self) -> bool:
        """Verifica si la matriz es diagonal."""
        if not self.is_square():
            return False

        for i in range(self.rows):
            for j in range(self.cols):
                if i != j and self.entries[i][j] != 0:
                    return False
        return True

    def get_row(self, index: int) -> 'Vector':
        """
        Obtiene una fila específica como vector.

        Args:
            index: Índice de la fila

        Returns:
            Vector con los elementos de la fila
        """
        if index < 0 or index >= self.rows:
            raise IndexError(
                f"Row index {index} is out of range for matrix with {self.rows} rows")

        return Vector(self.entries[index][:])  # Create copy to avoid mutation

    def get_column(self, index: int) -> 'Vector':
        """
        Get a specific Column as a Vector

        Args:
            index: Column Index

        Returns:
            Vector with the column elements
        """
        if index < 0 or index >= self.cols:
            raise IndexError(
                f"Column index {index} is out of range for matrix with {self.cols} columns")

        return Vector([self.entries[i][index] for i in range(self.rows)])

# =============================================================================
# VECTOR FUNCTIONS
# =============================================================================


def dot_product(v1: Vector, v2: Vector) -> float:
    """
    Calculates the dot product between two vectors.

    Args:
        v1: First vector
        v2: Second vector

    Returns:
        The dot product as a number
    """
    if len(v1) != len(v2):
        raise ValueError(
            "Vectors must have the same dimension for dot product")

    return sum(v1[i] * v2[i] for i in range(len(v1)))


def magnitude(v: Vector) -> float:
    """
    Calculates the magnitude (norm) of a vector.

    Args:
        v: The vector

    Returns:
        The magnitude of the vector
    """
    return math.sqrt(sum(component ** 2 for component in v))


def normalize(v: Vector) -> Vector:
    """
    Normalizes a vector (converts it to a unit vector).

    Args:
        v: The vector to normalize

    Returns:
        A new normalized vector
    """
    mag = magnitude(v)
    if mag == 0:
        raise ValueError("Cannot normalize a zero vector")

    return Vector([component / mag for component in v])


def cross_product(v1: Vector, v2: Vector) -> Vector:
    """
    Calculates the cross product between two 3D vectors.

    Args:
        v1: First vector
        v2: Second vector

    Returns:
        A new vector resulting from the cross product
    """
    if len(v1) != 3 or len(v2) != 3:
        raise ValueError("Cross product is only defined for 3D vectors")

    x = v1[1] * v2[2] - v1[2] * v2[1]
    y = v1[2] * v2[0] - v1[0] * v2[2]
    z = v1[0] * v2[1] - v1[1] * v2[0]

    return Vector([x, y, z])


def angle_between(v1: Vector, v2: Vector) -> float:
    """
    Calculates the angle between two vectors.

    Args:
        v1: First vector
        v2: Second vector

    Returns:
        The angle in radians
    """
    if len(v1) != len(v2):
        raise ValueError("Vectors must have the same dimension")

    dot_prod = dot_product(v1, v2)
    mag_v1 = magnitude(v1)
    mag_v2 = magnitude(v2)

    if mag_v1 == 0 or mag_v2 == 0:
        raise ValueError("Cannot calculate angle with zero vector")

    # Clamp the value to avoid numerical errors in math.acos
    cos_angle = dot_prod / (mag_v1 * mag_v2)
    cos_angle = max(-1.0, min(1.0, cos_angle))

    return math.math.acos(cos_angle)


# =============================================================================
# MATRIX FUNCTIONS
# =============================================================================

def scale(matrix: Matrix, scalar: Union[int, float]) -> Matrix:
    """
    Multiplies a matrix by a scalar.

    Args:
        matrix: The matrix
        scalar: The scalar

    Returns:
        A new scaled matrix
    """
    return matrix * scalar


def add(m1: Matrix, m2: Matrix) -> Matrix:
    """
    Adds two matrices.

    Args:
        m1: First matrix
        m2: Second matrix

    Returns:
        A new matrix resulting from the addition
    """
    return m1 + m2


def subtract(m1: Matrix, m2: Matrix) -> Matrix:
    """
    Subtracts two matrices.

    Args:
        m1: First matrix
        m2: Second matrix

    Returns:
        A new matrix resulting from the subtraction
    """
    return m1 - m2


def vector_multiply(matrix: Matrix, vector: Vector) -> Vector:
    """
    Multiplies a matrix by a vector.

    Args:
        matrix: The matrix
        vector: The vector

    Returns:
        A new vector resulting from the multiplication
    """
    return matrix * vector


def matrix_multiply(m1: Matrix, m2: Matrix) -> Matrix:
    """
    Multiplies two matrices.

    Args:
        m1: First matrix
        m2: Second matrix

    Returns:
        A new matrix resulting from the multiplication
    """
    return m1 * m2


def transpose(matrix: Matrix) -> Matrix:
    """
    Calculates the transpose of a matrix.

    Args:
        matrix: The matrix

    Returns:
        A new transposed matrix
    """
    return matrix.T


def determinant(matrix: Matrix) -> Union[int, float]:
    """
    Calculates the determinant of a square matrix.

    Args:
        matrix: The square matrix

    Returns:
        The determinant
    """
    return matrix.determinant


def inverse(matrix: Matrix) -> Matrix:
    """
    Calculates the inverse matrix.

    Args:
        matrix: The square invertible matrix

    Returns:
        A new inverse matrix
    """
    return matrix.inverse


def identity_matrix(size: int) -> Matrix:
    """
    Creates an identity matrix of specified size.

    Args:
        size: The size of the matrix (size x size)

    Returns:
        A new identity matrix
    """
    if size <= 0:
        raise ValueError("Matrix size must be positive")

    entries = [[1 if i == j else 0 for j in range(size)] for i in range(size)]
    return Matrix(entries)


def zeros_matrix(rows: int, columns: int) -> Matrix:
    """
    Creates a matrix of zeros with the specified dimensions.

    Args:
        rows: Number of rows
        columns: Number of columns

    Returns:
        A new matrix filled with zeros
    """
    if rows <= 0 or columns <= 0:
        raise ValueError("Matrix dimensions must be positive")

    entries = [[0 for _ in range(columns)] for _ in range(rows)]
    return Matrix(entries)


def ones_matrix(rows: int, columns: int) -> Matrix:
    """
    Creates a matrix of ones with the specified dimensions.

    Args:
        rows: Number of rows
        columns: Number of columns

    Returns:
        A new matrix filled with ones
    """
    if rows <= 0 or columns <= 0:
        raise ValueError("Matrix dimensions must be positive")

    entries = [[1 for _ in range(columns)] for _ in range(rows)]
    return Matrix(entries)
