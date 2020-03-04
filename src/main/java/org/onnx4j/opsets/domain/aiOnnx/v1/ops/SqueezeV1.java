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

import java.util.Arrays;
import java.util.List;

import org.onnx4j.Inputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.model.graph.node.attributes.IntsAttribute;
import org.onnx4j.opsets.domain.aiOnnx.v1.AiOnnxOperatorV1;
import org.onnx4j.opsets.operator.Field;
import org.onnx4j.opsets.operator.Field.TypeConstraint;
import org.onnx4j.opsets.operator.OperatorInputs;
import org.onnx4j.opsets.operator.fields.AttributeField;
import org.onnx4j.opsets.operator.fields.InputField;
import org.onnx4j.opsets.operator.output.SingleOperatorOutputs;
import org.onnx4j.tensor.DataType;

import com.google.common.collect.Lists;

/**
 * Squeeze Operator v1
 * 
 * <p>
 * Remove single-dimensional entries from the shape of a tensor. Takes a
 * parameter axes with a list of axes to squeeze. If axes is not provided, all
 * the single dimensions will be removed from the shape. If an axis is selected
 * with shape entry not equal to one, an error is raised.
 * 
 * @author HarryLee {@literal <formaten@qq.com>}
 * @version 1
 * @since Version 1 of the default ONNX operator set
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Changelog.md#Squeeze-1">
 *      ONNX.Changelog.md</a>
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Operators.md#Squeeze">
 *      ONNX.Operators.md</a>
 */
public interface SqueezeV1 extends AiOnnxOperatorV1 {

	public static final String OP_TYPE = "Squeeze";

	/**
	 * Constrain input and output types to all tensor types.
	 */
	public static final TypeConstraint TPYE_CONSTRAINT_T = new TypeConstraint(DataType.allTypes());

	/**
	 * Executes operator
	 * 
	 * @param data
	 *            Tensors with at least max(dims) dimensions.
	 * @param axes
	 *            List of integers indicating the dimensions to squeeze.
	 *            Negative value means counting dimensions from the back.
	 *            Accepted range is [-r, r-1] where r = rank(data).
	 * 
	 * @return Reshaped tensor with same data as input.
	 */
	// public abstract T_TENSOR squeeze(T_TENSOR data, List<Long> axes);

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
	class SqueezeInputsV1<T_TENSOR> extends OperatorInputs<T_TENSOR> {

		public static final String ATTR_AXES = "axes";

		/**
		 * List of non-negative integers, indicate the dimensions to squeeze.
		 */
		protected Field<List<Long>> axesField;

		/**
		 * Tensors with at least max(dims) dimensions.
		 */
		protected Field<T_TENSOR> dataField;

		public SqueezeInputsV1(Node node, Inputs inputs) {
			super(node, inputs);

			dataField = new InputField<T_TENSOR>(this, TPYE_CONSTRAINT_T, super.inputArray[0]);

			//
			// int (default is 1)
			//
			axesField = new AttributeField<List<Long>>(super.attrs, ATTR_AXES, IntsAttribute.class,
					Lists.newLinkedList(), false);

			for (Long axis : this.axesField.getData()) {
				if (axis < 0) {
					throw new IllegalArgumentException(
							String.format("The list of axes%s can not contains negative integers.", Arrays.deepToString(
									this.axesField.getData().toArray(new Long[this.axesField.getData().size()]))));
				}
			}
		}

		public T_TENSOR getData() {
			return dataField.getData();
		}

		public List<Long> getAxes() {
			return axesField.getData();
		}

	}

	/**
	 * Outputs for operator execution (forward & backward)
	 * 
	 * @param <T_TENSOR>
	 *            The backend tensor object.
	 */
	class SqueezeOutputV1<T_TENSOR> extends SingleOperatorOutputs<T_TENSOR> {

		public SqueezeOutputV1(T_TENSOR output) {
			super(output);
		}

		@Override
		public TypeConstraint getTypeConstraint() {
			return TPYE_CONSTRAINT_T;
		}

	}

}