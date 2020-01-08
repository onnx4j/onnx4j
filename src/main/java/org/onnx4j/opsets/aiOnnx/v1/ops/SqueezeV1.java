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
package org.onnx4j.opsets.aiOnnx.v1.ops;

import java.util.Arrays;
import java.util.List;

import org.onnx4j.Inputs;
import org.onnx4j.Inputs.Input;
import org.onnx4j.Outputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.model.graph.node.Attributes;
import org.onnx4j.model.graph.node.attributes.IntsAttribute;
import org.onnx4j.opsets.aiOnnx.v1.AiOnnxOperatorV1;
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
public interface SqueezeV1<T_TENSOR> extends AiOnnxOperatorV1 {

	public static final String OP_TYPE = "Squeeze";

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
	public abstract T_TENSOR squeeze(T_TENSOR data, List<Long> axes);

	@Override
	public default DataType[] getTypeConstraints() {
		return DataType.floatTypes();
	}

	@Override
	public default String getOpType() {
		return OP_TYPE;
	}

	class SqueezeInputsV1<T_TENSOR> {

		//
		// list of ints
		//
		public static final String ATTR_AXES = "axes";

		protected T_TENSOR data;
		protected List<Long> axes;

		public SqueezeInputsV1(Node node, Inputs inputs) {
			super();

			Attributes attrs = node.getAttrs();
			Input[] inputArray = inputs.get();

			this.data = inputArray[0].getTensor();
			this.axes = attrs.getAttrValue(ATTR_AXES, IntsAttribute.class, Lists.newLinkedList());

			for (Long axis : this.axes) {
				if (axis < 0) {
					throw new IllegalArgumentException(
							String.format("The list of axes%s can not contains negative integers.",
									Arrays.deepToString(this.axes.toArray(new Long[this.axes.size()]))));
				}
			}
		}

		public T_TENSOR getData() {
			return data;
		}

		public List<Long> getAxes() {
			return axes;
		}

	}

	@Override
	public default Outputs forward(Node node, Inputs inputs) {
		SqueezeInputsV1<T_TENSOR> operatorInputs = new SqueezeInputsV1<T_TENSOR>(node, inputs);
		return Outputs.wrap(node, this.squeeze(operatorInputs.getData(), operatorInputs.getAxes()));
	}

}