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
 * ArgMax Operator v1
 * 
 * <p>
 * Computes the indices of the max elements of the input tensor's element along
 * the provided axis. The resulted tensor has the same rank as the input if
 * keepdims equal 1. If keepdims equal 0, then the resulted tensor have the
 * reduced dimension pruned. The type of the output tensor is integer.
 * 
 * @author HarryLee {@literal <formaten@qq.com>}
 * @version 1
 * @since Version 1 of the default ONNX operator set
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Changelog.md#ArgMax-1">
 *      ONNX.Changelog.md</a>
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Operators.md#ArgMax">ONNX
 *      .Operators.md</a>
 */
public interface ArgMaxV1 extends AiOnnxOperatorV1 {

	public static final String OP_TYPE = "ArgMax";

	/**
	 * Constrain input and output types to all numeric tensors.
	 */
	public static final TypeConstraint TPYE_CONSTRAINT_T1 = new TypeConstraint(DataType.numericTypes());

	/**
	 * Constrain output to int64 tensor.
	 */
	public static final TypeConstraint TPYE_CONSTRAINT_T2 = new TypeConstraint(DataType.INT64);

	/**
	 * Executes operator
	 * 
	 * @param data
	 *            An input tensor.
	 * @param axis
	 *            The axis in which to compute the arg indices.
	 * @param keepdims
	 *            Keep the reduced dimension or not, default 1 mean keep reduced
	 *            dimension.
	 * @return Reduced output tensor with integer data type.
	 */
	/*public abstract T_TENSOR argmax(T_TENSOR data, Long axis, Long keepdims);*/

	@Override
	public default OperatorStatus getStatus() {
		return OperatorStatus.STABLE;
	}

	@Override
	public default String getOpType() {
		return OP_TYPE;
	}

	class ArgMaxInputsV1<T_TENSOR> extends OperatorInputs<T_TENSOR> {

		//
		// The axis in which to compute the arg indices.
		//
		public static final String ATTR_AXIS = "axis";

		//
		// Keep the reduced dimension or not, default 1 mean keep reduced
		// dimension.
		//
		public static final String ATTR_KEEPDIMS = "keepdims";

		protected Field<T_TENSOR> dataField = new InputField<T_TENSOR>(this, TPYE_CONSTRAINT_T1, super.inputArray[0]);

		protected Field<Long> axisField = new AttributeField<Long>(super.attrs, ATTR_AXIS, IntAttribute.class, 0L,
				false);

		protected Field<Long> keepdimsField = new AttributeField<Long>(super.attrs, ATTR_KEEPDIMS, IntAttribute.class,
				1L, false);

		public ArgMaxInputsV1(Node node, Inputs inputs) {
			super(node, inputs);
		}

		public T_TENSOR getData() {
			return dataField.getData();
		}

		public Long getAxis() {
			return axisField.getData();
		}

		public Long getKeepdims() {
			return keepdimsField.getData();
		}

	}

	/**
	 * Outputs for operator execution (forward & backward)
	 *
	 * @param <T_TENSOR>
	 *            The backend tensor object.
	 */
	class ArgMaxOutputV1<T_TENSOR> extends SingleOperatorOutputs<T_TENSOR> {

		public ArgMaxOutputV1(T_TENSOR output) {
			super(output);
		}

		@Override
		public TypeConstraint getTypeConstraint() {
			return TPYE_CONSTRAINT_T2;
		}

	}

}