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
package org.onnx4j.opsets.domain.aiOnnx.v1.ops;

import org.onnx4j.Inputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.model.graph.node.attributes.IntAttribute;
import org.onnx4j.opsets.domain.aiOnnx.v1.AiOnnxOperatorV1;
import org.onnx4j.opsets.operator.Field;
import org.onnx4j.opsets.operator.Field.TypeConstraint;
import org.onnx4j.opsets.operator.OperatorInputs;
import org.onnx4j.opsets.operator.fields.AttributeField;
import org.onnx4j.opsets.operator.fields.InputField;
import org.onnx4j.opsets.operator.output.SingleOperatorOutputs;
import org.onnx4j.tensor.DataType;

/**
 * Softmax Operator v1
 * 
 * <p>
 * The operator computes the softmax (normalized exponential) values for each
 * layer in the batch of the given input. The input is a 2-D tensor (Tensor) of
 * size (batch_size x input_feature_dimensions). The output tensor has the same
 * shape and contains the softmax values of the corresponding input.
 * 
 * <p>
 * Input does not need to explicitly be a 2D vector; rather, it will be coerced
 * into one. For an arbitrary n-dimensional tensor input \in [a_0, a_1, ...,
 * a_{k-1}, a_k, ..., a_{n-1}] and k is the axis provided, then input will be
 * coerced into a 2-dimensional tensor with dimensions [a_0 * ... * a_{k-1}, a_k
 * * ... * a_{n-1}]. For the default case where axis=1, this means the input
 * tensor will be coerced into a 2D tensor of dimensions [a_0, a_1 * ... *
 * a_{n-1}], where a_0 is often the batch size. In this situation, we must have
 * a_0 = N and a_1 * ... * a_{n-1} = D. Each of these dimensions must be matched
 * correctly, or else the operator will throw errors.
 * 
 * @author HarryLee {@literal <formaten@qq.com>}
 * @version 1
 * @since Version 1 of the default ONNX operator set
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Changelog.md#Softmax-1">
 *      ONNX. Changelog.md</a>
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Operators.md#Softmax">
 *      ONNX. Operators.md</a>
 */
public interface SoftmaxV1 extends AiOnnxOperatorV1 {

	public static final String OP_TYPE = "Softmax";

	/**
	 * Constrain input and output types to float tensors.
	 */
	public static final TypeConstraint TPYE_CONSTRAINT_T = new TypeConstraint(DataType.floatTypes());

	/**
	 * Executes operator
	 * 
	 * @param input
	 *            The input tensor that's coerced into a 2D matrix of size (NxD)
	 *            as described above.
	 * @param axis
	 *            Describes the axis of the inputs when coerced to 2D; defaults
	 *            to one because the 0th axis most likely describes the
	 *            batch_size.
	 * @return The output values with the same shape as input tensor (the
	 *         original size without coercion).
	 */
	// public abstract T_TENSOR softmax(T_TENSOR input, Long axis);

	@Override
	public default OperatorStatus getStatus() {
		return OperatorStatus.STABLE;
	}

	@Override
	public default String getOpType() {
		return OP_TYPE;
	}

	/**
	 * Inputs for operator execution (forward & backward)
	 *
	 * @param <T_TENSOR>
	 *            The backend tensor object.
	 */
	class SoftmaxInputsV1<T_TENSOR> extends OperatorInputs<T_TENSOR> {

		public static final String ATTR_AXIS = "axis";

		/**
		 * Describes the axis of the inputs when coerced to 2D; defaults to one
		 * because the 0th axis most likely describes the batch_size
		 */
		protected Field<Long> axisField;

		/**
		 * The input tensor that's coerced into a 2D matrix of size (NxD) as
		 * described above.
		 */
		protected Field<T_TENSOR> inputField;

		public SoftmaxInputsV1(Node node, Inputs inputs) {
			super(node, inputs);

			inputField = new InputField<T_TENSOR>(this, TPYE_CONSTRAINT_T, super.inputArray[0]);

			//
			// int (default is 1)
			//
			axisField = new AttributeField<Long>(super.attrs, ATTR_AXIS, IntAttribute.class, 1L, false);

			if (this.axisField.getData() < 0L)
				throw new IllegalArgumentException(
						String.format("Attribute named \"%s\" can not be a negative value(Axis=%s).", ATTR_AXIS,
								this.axisField.getData()));
		}

		public T_TENSOR getInput() {
			return inputField.getData();
		}

		public Long getAxis() {
			return axisField.getData();
		}

	}

	/**
	 * Outputs for operator execution (forward & backward)
	 * 
	 * @param <T_TENSOR>
	 *            The backend tensor object.
	 */
	class SoftmaxOutputV1<T_TENSOR> extends SingleOperatorOutputs<T_TENSOR> {

		public SoftmaxOutputV1(T_TENSOR output) {
			super(output);
		}

		@Override
		public TypeConstraint getTypeConstraint() {
			return TPYE_CONSTRAINT_T;
		}

	}

}