/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.onnx4j.tensor;

import java.util.Arrays;
import java.util.List;

import org.onnx4j.prototypes.OnnxProto3.TensorShapeProto;

import com.google.common.primitives.Longs;

public class Shape {

	private long[] shape;

	public static Shape toShape(TensorShapeProto shapeProto) {
		long[] shape = new long[shapeProto.getDimCount()];
		for (int n = 0; n < shape.length; n++) {
			TensorShapeProto.Dimension dimension = shapeProto.getDimList().get(n);
			assert dimension != null;
			shape[n] = dimension.getDimValue();
		}

		return Shape.create(shape);
	}

	public static Shape create(long... shape) {
		return new Shape(shape);
	}

	public static Shape create(List<Long> shape) {
		return new Shape(Longs.toArray(shape));
	}

	private Shape(long... shape) {
		this.shape = shape;
	}

	public long numElements() {
		long numElements = 1;
		for (long numElementsInThisDim : this.shape) {
			numElements *= numElementsInThisDim;
		}
		return numElements;
	}

	public int dims() {
		return this.shape.length;
	}

	public long get(int i) {
		assert this.shape.length > i;
		return this.shape[i];
	}

	public long[] toArray() {
		return this.shape;
	}

	@Override
	public String toString() {
		return Arrays.deepToString(Longs.asList(this.shape).toArray(new Long[this.shape.length]));
	}

	@Override
	public boolean equals(Object o) {
		if (o != null && o instanceof Shape) {
			if (((Shape) o).shape.length != this.shape.length)
				return false;

			for (int n = 0; n < this.shape.length; n++) {
				//
				// Batch dimension: Zero means any batch
				//
				if (n == 0 && this.shape[n] == 0) {
					continue;
				}

				if (((Shape) o).shape[n] != this.shape[n])
					return false;
			}

			return true;
			// return Arrays.equals(((Shape) o).shape, this.shape);
		}
		return false;
	}

}