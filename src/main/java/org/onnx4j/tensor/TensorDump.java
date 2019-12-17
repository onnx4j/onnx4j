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

import java.nio.ByteBuffer;
import java.util.Arrays;

import org.onnx4j.Tensor;

public final class TensorDump {

	public static String dump(Tensor tensor) {
		long[] shape = tensor.getShape();
		Integer[] shapeInInt = new Integer[shape.length];
		for (int n = 0; n < shape.length; n++) {
			shapeInInt[n] = (int) shape[n];
		}
		ByteBuffer dataBuffer = tensor.getData();
		return "Tensor" + Arrays.deepToString(shapeInInt) + " = \n" + TensorDump.dump(tensor, dataBuffer, 0, shapeInInt);
	}

	private static StringBuffer dump(Tensor tensor, ByteBuffer dataBuffer, int axis, Integer... coords) {
		StringBuffer sb = new StringBuffer();
		for (int n = 0; n < axis; n++)
			sb.append("\t");
		sb.append("[");
		if (axis != coords.length - 1)
			sb.append("\n");
		for (int n = 0; n < coords[axis]; n++) {
			Integer[] nextCursor = coords.clone();
			nextCursor[axis] = n;

			if (axis < coords.length - 1) {
				sb.append(TensorDump.dump(tensor, dataBuffer, axis + 1, nextCursor));
			}

			if (axis == coords.length - 1) {
				if (n > 0)
					sb.append(",\t");
				sb.append(TensorDump.dump(tensor, dataBuffer, nextCursor));
			}
		}
		if (axis != coords.length - 1)
			for (int n = 0; n < axis; n++)
				sb.append("\t");
		sb.append("]");
		if (axis != 0)
			sb.append(",");
		sb.append("\n");

		return sb;
	}

	private static String dump(Tensor tensor, ByteBuffer dataBuffer, Integer... coords) {
		int position = 0;
		for (int n = 0; n < coords.length; n++) {
			position += coords[n] * TensorDump.nbElements(tensor, n);
		}

		if (DataType.INT16.equals(tensor.getDataType())) {
			return String.valueOf(dataBuffer.getShort(position * DataType.INT16.getUnitSize()));
		} else if (DataType.INT32.equals(tensor.getDataType())) {
			return String.valueOf(dataBuffer.getInt(position * DataType.INT32.getUnitSize()));
		} else if (DataType.INT64.equals(tensor.getDataType())) {
			return String.valueOf(dataBuffer.getLong(position * DataType.INT64.getUnitSize()));
		} else if (DataType.DOUBLE.equals(tensor.getDataType())) {
			return String.valueOf(dataBuffer.getDouble(position * DataType.DOUBLE.getUnitSize()));
		} else if (DataType.FLOAT.equals(tensor.getDataType())) {
			return String.valueOf(dataBuffer.getFloat(position * DataType.FLOAT.getUnitSize()));
		} else {
			throw new UnsupportedOperationException(
					String.format("%s not supported to be dump.", tensor.getDataType()));
		}
	}

	private static int nbElements(Tensor tensor, int axis) {
		int nbElements = 1;
		long[] shape = tensor.getShape();
		for (int n = axis + 1; n < shape.length; n++) {
			nbElements *= (int) shape[n];
		}
		return nbElements;
	}

}