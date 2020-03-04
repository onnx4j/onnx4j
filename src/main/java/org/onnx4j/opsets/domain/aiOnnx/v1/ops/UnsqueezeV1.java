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
 * Unsqueeze Operator v1
 * 
 * <p>
 * Insert single-dimensional entries to the shape of a tensor. Takes one
 * required argument axes, a list of dimensions that will be inserted. Dimension
 * indices in axes are as seen in the output tensor. For example: Given a tensor
 * such that tensor with shape [3, 4, 5], then Unsqueeze(tensor, axes=[0, 4])
 * has shape [1, 3, 4, 5, 1].
 * 
 * @author HarryLee {@literal <formaten@qq.com>}
 * @version 1
 * @since Version 1 of the default ONNX operator set
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Changelog.md#Unsqueeze-1">
 *      ONNX.Changelog.md</a>
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Operators.md#Unsqueeze">
 *      ONNX.Operators.md</a>
 */
public interface UnsqueezeV1 extends AiOnnxOperatorV1 {

	public static final String OP_TYPE = "Unsqueeze";

	/**
	 * Constrain input and output types to all tensor types.
	 */
	public static final TypeConstraint TPYE_CONSTRAINT_T = new TypeConstraint(DataType.allTypes());

	/**
	 * Executes operator
	 * 
	 * @param data
	 *            Original tensor.
	 * @param axes
	 *            List of non-negative integers, indicate the dimensions to be
	 *            inserted.
	 * 
	 * @return Reshaped tensor with same data as input.
	 */
	// public abstract T_TENSOR unsqueeze(T_TENSOR data, List<Long> axes);

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
	class UnsqueezeInputsV1<T_TENSOR> extends OperatorInputs<T_TENSOR> {

		public static final String ATTR_AXES = "axes";

		/**
		 * List of non-negative integers, indicate the dimensions to be
		 * inserted.
		 */
		protected Field<List<Long>> axesField;

		/**
		 * Original tensor.
		 */
		protected Field<T_TENSOR> dataField;

		public UnsqueezeInputsV1(Node node, Inputs inputs) {
			super(node, inputs);

			dataField = new InputField<T_TENSOR>(this, TPYE_CONSTRAINT_T, super.inputArray[0]);

			//
			// int (default is 1)
			//
			axesField = new AttributeField<List<Long>>(super.attrs, ATTR_AXES, IntsAttribute.class,
					Lists.newLinkedList(), true);

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
	class UnsqueezeOutputV1<T_TENSOR> extends SingleOperatorOutputs<T_TENSOR> {

		public UnsqueezeOutputV1(T_TENSOR output) {
			super(output);
		}

		@Override
		public TypeConstraint getTypeConstraint() {
			return TPYE_CONSTRAINT_T;
		}

	}

}