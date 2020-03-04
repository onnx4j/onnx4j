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
import org.onnx4j.model.graph.node.attributes.IntAttribute;
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
 * ReduceMax Operator v1
 * 
 * <p>
 * Computes the max of the input tensor's element along the provided axes. The
 * resulted tensor has the same rank as the input if keepdims equal 1. If
 * keepdims equal 0, then the resulted tensor have the reduced dimension pruned.
 * 
 * <p>
 * The above behavior is similar to numpy, with the exception that numpy default
 * keepdims to False instead of True.
 * 
 * @author HarryLee {@literal <formaten@qq.com>}
 * @version 1
 * @since Version 1 of the default ONNX operator set
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Changelog.md#ReduceMax-1">
 *      ONNX.Changelog.md</a>
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceMax">
 *      ONNX.Operators.md</a>
 */
public interface ReduceMaxV1 extends AiOnnxOperatorV1 {

	public static final String OP_TYPE = "ReduceMax";

	/**
	 * Constrain input and output types to high-precision numeric tensors.
	 */
	public static final TypeConstraint TPYE_CONSTRAINT_T = new TypeConstraint(DataType.highPrecisionNumeric());

	/**
	 * Executes operator
	 * 
	 * @param data
	 *            An input tensor.
	 * @param axes
	 *            A list of integers, along which to reduce. The default is to
	 *            reduce over all the dimensions of the input tensor.
	 * @param keepdims
	 *            Keep the reduced dimension or not, default 1 mean keep reduced
	 *            dimension.
	 * 
	 * @return Reduced output tensor.
	 */
	//public abstract T_TENSOR reduceMax(T_TENSOR data, List<Long> axes, Long keepdims);

	@Override
	public default String getOpType() {
		return OP_TYPE;
	}

	@Override
	public default OperatorStatus getStatus() {
		return OperatorStatus.STABLE;
	}

	/**
	 * Inputs for operator execution (forward & backward)
	 *
	 * @param <T_TENSOR>
	 *            The backend tensor object.
	 */
	class ReduceMaxInputsV1<T_TENSOR> extends OperatorInputs<T_TENSOR> {

		//
		// list of ints
		//
		public static final String ATTR_AXES = "axes";

		//
		// int (default is 1)
		//
		public static final String ATTR_KEEPDIMS = "keepdims";

		protected Field<T_TENSOR> dataField;

		protected Field<List<Long>> axesField;

		protected Field<Long> keepdimsField;

		public ReduceMaxInputsV1(Node node, Inputs inputs) {
			super(node, inputs);

			this.dataField = new InputField<T_TENSOR>(this, TPYE_CONSTRAINT_T, super.inputArray[0]);
			this.axesField = new AttributeField<List<Long>>(super.attrs, ATTR_AXES, IntsAttribute.class,
					Lists.newLinkedList(), false);
			this.keepdimsField = new AttributeField<Long>(super.attrs, ATTR_KEEPDIMS, IntAttribute.class, 1L, false);

			for (Long axis : axesField.getData()) {
				if (axis < 0) {
					throw new IllegalArgumentException(
							String.format("The list of axes%s can not contains negative values.", Arrays.deepToString(
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

		public Long getKeepdims() {
			return keepdimsField.getData();
		}

	}

	/**
	 * Outputs for operator execution (forward & backward) s
	 * 
	 * @param <T_TENSOR>
	 *            The backend tensor object.
	 */
	class ReduceMaxOutputV1<T_TENSOR> extends SingleOperatorOutputs<T_TENSOR> {

		public ReduceMaxOutputV1(T_TENSOR output) {
			super(output);
		}

		@Override
		public TypeConstraint getTypeConstraint() {
			return TPYE_CONSTRAINT_T;
		}

	}

}