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
public interface UnsqueezeV1<T_TENSOR> extends AiOnnxOperatorV1 {

	public static final String OP_TYPE = "Unsqueeze";

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
	public abstract T_TENSOR unsqueeze(T_TENSOR data, List<Long> axes);

	@Override
	public default DataType[] getTypeConstraints() {
		return DataType.floatTypes();
	}

	@Override
	public default String getOpType() {
		return OP_TYPE;
	}

	class UnsqueezeInputsV1<T_TENSOR> {

		//
		// list of ints
		//
		public static final String ATTR_AXES = "axes";

		protected T_TENSOR data;
		protected List<Long> axes;

		public UnsqueezeInputsV1(Node node, Inputs inputs) {
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
		UnsqueezeInputsV1<T_TENSOR> operatorInputs = new UnsqueezeInputsV1<T_TENSOR>(node, inputs);
		return Outputs.wrap(node, this.unsqueeze(operatorInputs.getData(), operatorInputs.getAxes()));
	}

}