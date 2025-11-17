# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the LM-Service project

import numpy as np
import pytest

try:
    from lm_service.apis.vllm.proxy import _encode_mm_data
except ImportError:
    pytest.skip(
        "vllm dependencies not available for integration test",
        allow_module_level=True,
    )


class TestEncodeMMData:
    """Test suite for the _encode_mm_data function."""

    def test_encode_single_numpy_array(self):
        """Test encoding a single numpy array."""
        # Create a simple numpy array
        img = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        mm_data = {"image": img}

        result = _encode_mm_data(mm_data)

        # Verify the structure
        assert "image" in result
        assert len(result["image"]) == 1

        encoded_img = result["image"][0]
        assert encoded_img["type"] == "ndarray"
        assert encoded_img["shape"] == (2, 3)
        assert encoded_img["dtype"] == "float32"

        # Verify the data can be reconstructed
        reconstructed = np.frombuffer(
            encoded_img["data"], dtype=np.float32
        ).reshape(encoded_img["shape"])
        np.testing.assert_array_equal(reconstructed, img)

    def test_encode_list_of_numpy_arrays(self):
        """Test encoding a list of numpy arrays."""
        img1 = np.array([1, 2, 3], dtype=np.int32)
        img2 = np.array([[10, 20], [30, 40]], dtype=np.float64)
        mm_data = {"image": [img1, img2]}

        result = _encode_mm_data(mm_data)

        # Verify the structure
        assert "image" in result
        assert len(result["image"]) == 2

        # Check first image
        encoded_img1 = result["image"][0]
        assert encoded_img1["type"] == "ndarray"
        assert encoded_img1["shape"] == (3,)
        assert encoded_img1["dtype"] == "int32"

        # Check second image
        encoded_img2 = result["image"][1]
        assert encoded_img2["type"] == "ndarray"
        assert encoded_img2["shape"] == (2, 2)
        assert encoded_img2["dtype"] == "float64"

        # Verify data reconstruction
        reconstructed1 = np.frombuffer(
            encoded_img1["data"], dtype=np.int32
        ).reshape(encoded_img1["shape"])
        reconstructed2 = np.frombuffer(
            encoded_img2["data"], dtype=np.float64
        ).reshape(encoded_img2["shape"])
        np.testing.assert_array_equal(reconstructed1, img1)
        np.testing.assert_array_equal(reconstructed2, img2)

    def test_encode_empty_image_list(self):
        """Test encoding when no images are provided."""
        mm_data = {"image": []}

        result = _encode_mm_data(mm_data)

        assert "image" in result
        assert result["image"] == []

    def test_encode_no_image_key(self):
        """Test encoding when 'image' key is not present."""
        mm_data = {"other_data": "some_value"}

        result = _encode_mm_data(mm_data)

        assert "image" in result
        assert result["image"] == []

    def test_encode_non_numpy_images_are_ignored(self):
        """Test that non-numpy array images are ignored."""
        img_array = np.array([1, 2, 3], dtype=np.int32)
        non_numpy_img = "not_an_array"
        mm_data = {"image": [img_array, non_numpy_img]}

        result = _encode_mm_data(mm_data)

        # Only the numpy array should be encoded
        assert "image" in result
        assert len(result["image"]) == 1

        encoded_img = result["image"][0]
        assert encoded_img["type"] == "ndarray"
        assert encoded_img["shape"] == (3,)
        assert encoded_img["dtype"] == "int32"

    def test_encode_mixed_numpy_and_non_numpy_images(self):
        """Test encoding with a mix of numpy arrays and other types."""
        img1 = np.array([[1, 2]], dtype=np.uint8)
        img2 = np.array([10, 20, 30], dtype=np.float32)
        mm_data = {"image": [img1, "string", img2, 123, None]}

        result = _encode_mm_data(mm_data)

        # Only numpy arrays should be encoded
        assert "image" in result
        assert len(result["image"]) == 2

        # Verify first numpy array
        encoded_img1 = result["image"][0]
        assert encoded_img1["type"] == "ndarray"
        assert encoded_img1["shape"] == (1, 2)
        assert encoded_img1["dtype"] == "uint8"

        # Verify second numpy array
        encoded_img2 = result["image"][1]
        assert encoded_img2["type"] == "ndarray"
        assert encoded_img2["shape"] == (3,)
        assert encoded_img2["dtype"] == "float32"

    def test_encode_single_non_list_numpy_array(self):
        """Test encoding when image is a single numpy array (not in a list)."""
        img = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.int16)
        mm_data = {"image": img}

        result = _encode_mm_data(mm_data)

        assert "image" in result
        assert len(result["image"]) == 1

        encoded_img = result["image"][0]
        assert encoded_img["type"] == "ndarray"
        assert encoded_img["shape"] == (1, 2, 3)
        assert encoded_img["dtype"] == "int16"

    def test_encode_single_non_list_non_numpy(self):
        """Test encoding when image is a single non-numpy object (not in a list)."""
        mm_data = {"image": "not_numpy"}

        result = _encode_mm_data(mm_data)

        assert "image" in result
        assert result["image"] == []

    def test_encode_different_numpy_dtypes(self):
        """Test encoding numpy arrays with different data types."""
        test_cases = [
            (np.array([1, 2]), np.int8),
            (np.array([1, 2]), np.int16),
            (np.array([1, 2]), np.int32),
            (np.array([1, 2]), np.int64),
            (np.array([1.0, 2.0]), np.float16),
            (np.array([1.0, 2.0]), np.float32),
            (np.array([1.0, 2.0]), np.float64),
            (np.array([True, False]), np.bool_),
        ]

        for base_array, dtype in test_cases:
            img = base_array.astype(dtype)
            mm_data = {"image": img}

            result = _encode_mm_data(mm_data)

            assert len(result["image"]) == 1
            encoded_img = result["image"][0]
            assert encoded_img["dtype"] == str(dtype)

            # Verify reconstruction
            reconstructed = np.frombuffer(
                encoded_img["data"], dtype=dtype
            ).reshape(encoded_img["shape"])
            np.testing.assert_array_equal(reconstructed, img)

    def test_encode_complex_numpy_array_shapes(self):
        """Test encoding numpy arrays with various complex shapes."""
        test_shapes = [
            (10,),  # 1D
            (5, 4),  # 2D
            (2, 3, 4),  # 3D
            (2, 2, 2, 2),  # 4D
            (1,),  # Single element
            (1, 1, 1),  # Single element in multiple dimensions
        ]

        for shape in test_shapes:
            img = np.arange(np.prod(shape)).reshape(shape).astype(np.float32)
            mm_data = {"image": img}

            result = _encode_mm_data(mm_data)

            assert len(result["image"]) == 1
            encoded_img = result["image"][0]
            assert encoded_img["shape"] == shape

            # Verify reconstruction
            reconstructed = np.frombuffer(
                encoded_img["data"], dtype=np.float32
            ).reshape(shape)
            np.testing.assert_array_equal(reconstructed, img)
