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
package org.onnx4j.opsets.domain.aiOnnx.v5.ops;

import java.util.List;

import org.onnx4j.Inputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.ReshapeV1;
import org.onnx4j.opsets.domain.aiOnnx.v5.AiOnnxOperatorV5;
import org.onnx4j.opsets.operator.Field;
import org.onnx4j.opsets.operator.Field.TypeConstraint;
import org.onnx4j.opsets.operator.fields.InputField;
import org.onnx4j.tensor.DataType;

/**
 * Reshape Operator v5
 * 
 * <p>
 * Operator Reshape the input tensor similar to numpy.reshape. It takes a tensor
 * as input and an argument shape. It outputs the reshaped tensor. At most one
 * dimension of the new shape can be -1. In this case, the value is inferred
 * from the size of the tensor and the remaining dimensions. A dimension could
 * also be 0, in which case the actual dimension value is unchanged (i.e. taken
 * from the input tensor).
 * 
 * @author HarryLee {@literal <formaten@qq.com>}
 * @version 4
 * @since Version 5 of the default ONNX operator set
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Changelog.md#Reshape-5">
 *      ONNX.Changelog.md</a>
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Operators.md#Reshape">
 *      ONNX .Operators.md</a>
 * @see ReshapeV1
 */
public interface ReshapeV5 extends ReshapeV1, AiOnnxOperatorV5 {

	/**
	 * Constrain input and output types to all tensor types.
	 */
	public static final TypeConstraint TPYE_CONSTRAINT_T = new TypeConstraint(DataType.allTypes());

	public static final TypeConstraint TPYE_CONSTRAINT_INT64 = new TypeConstraint(DataType.INT64);

	/**
	 * Reshape the input tensor similar to numpy.reshape. It takes a tensor as
	 * input and an argument shape. It outputs the reshaped tensor. At most one
	 * dimension of the new shape can be -1. In this case, the value is inferred
	 * from the size of the tensor and the remaining dimensions. A dimension
	 * could also be 0, in which case the actual dimension value is unchanged
	 * (i.e. taken from the input tensor).
	 * 
	 * @param data
	 *            An input tensor
	 * @param shape
	 *            New shape
	 * @param consumedInputs
	 *            legacy optimization attribute
	 * @return Reshaped data
	 */
	//public abstract T_TENSOR reshape(T_TENSOR data, T_TENSOR shape);

	class ReshapeInputsV5<T_TENSOR> extends ReshapeInputsV1<T_TENSOR> {

		private Field<T_TENSOR> shapeTensorField;

		public ReshapeInputsV5(Node node, Inputs inputs) {
			super(node, inputs);

			this.shapeTensorField = new InputField<T_TENSOR>(this, TPYE_CONSTRAINT_INT64, super.inputArray[1]);
		}

		public T_TENSOR getShapeTensor() {
			return shapeTensorField.getData();
		}

		public List<Long> getShape() {
			throw new UnsupportedOperationException(String.format("Attribute named \"%s\" has deprecated", ATTR_SHAPE));
		}

		public List<Long> getConsumedInputs() {
			throw new UnsupportedOperationException(
					String.format("Attribute named \"%s\" has deprecated", ATTR_CONSUMED_INPUTS));
		}

	}

	/**
	 * Outputs for operator execution (forward & backward)
	 *
	 * @param <T_TENSOR>
	 *            The backend tensor object.
	 */
	class ReshapeOutputV5<T_TENSOR> extends ReshapeOutputV1<T_TENSOR> {

		public ReshapeOutputV5(T_TENSOR output) {
			super(output);
		}

		@Override
		public TypeConstraint getTypeConstraint() {
			return TPYE_CONSTRAINT_T;
		}

	}

}