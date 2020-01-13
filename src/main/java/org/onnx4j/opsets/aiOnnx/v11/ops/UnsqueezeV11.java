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
package org.onnx4j.opsets.aiOnnx.v11.ops;

import java.util.List;

import org.onnx4j.Inputs;
import org.onnx4j.Inputs.Input;
import org.onnx4j.Outputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.model.graph.node.Attributes;
import org.onnx4j.model.graph.node.attributes.IntsAttribute;
import org.onnx4j.opsets.aiOnnx.v1.ops.UnsqueezeV1;
import org.onnx4j.opsets.aiOnnx.v11.AiOnnxOperatorV11;

import com.google.common.collect.Lists;

/**
 * Unsqueeze Operator v11
 * 
 * <p>
 * Insert single-dimensional entries to the shape of an input tensor (data).
 * Takes one required argument axes - which contains a list of dimension indices
 * and this operator will insert a dimension of value 1 into the corresponding
 * index of the output tensor (expanded).
 * 
 * <p>
 * For example: Given an input tensor (data) of shape [3, 4, 5], then
 * Unsqueeze(data, axes=[0, 4]) outputs a tensor (expanded) containing same data
 * as data but with shape [1, 3, 4, 5, 1].
 * 
 * <p>
 * The attribute axes should not contain any duplicate entries. It is an error
 * if it contains duplicates. The rank of the output tensor (output_rank) is the
 * rank of the input tensor (data) plus the number of values in axes. Each value
 * in axes should be within the (inclusive) range [-output_rank , output_rank -
 * 1]. The order of values in axes does not matter and can come in any order.
 * 
 * @author HarryLee {@literal <formaten@qq.com>}
 * @version 11
 * @since Version 1 of the default ONNX operator set
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Changelog.md#Unsqueeze-11">
 *      ONNX.Changelog.md</a>
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Operators.md#Unsqueeze">
 *      ONNX.Operators.md</a>
 */
public interface UnsqueezeV11<T_TENSOR> extends UnsqueezeV1<T_TENSOR>, AiOnnxOperatorV11 {

	class UnsqueezeInputsV11<T_TENSOR> extends UnsqueezeInputsV1<T_TENSOR> {

		public UnsqueezeInputsV11(Node node, Inputs inputs) {
			super(node, inputs);

			Attributes attrs = node.getAttrs();
			Input[] inputArray = inputs.get();

			super.data = inputArray[0].getTensor();
			super.axes = attrs.getAttrValue(ATTR_AXES, IntsAttribute.class, Lists.newLinkedList());
		}

	}

	/**
	 * Executes operator
	 * 
	 * @param data
	 *            Original tensor.
	 * @param axes
	 *            List of integers indicating the dimensions to be inserted.
	 *            Negative value means counting dimensions from the back.
	 *            Accepted range is [-r, r-1] where r = rank(expanded).
	 * 
	 * @return Reshaped tensor with same data as input.
	 */
	public abstract T_TENSOR unsqueeze(T_TENSOR data, List<Long> axes);

	@Override
	public default Outputs forward(Node node, Inputs inputs) {
		UnsqueezeInputsV11<T_TENSOR> operatorInputs = new UnsqueezeInputsV11<T_TENSOR>(node, inputs);
		return Outputs.wrap(node, this.unsqueeze(operatorInputs.getData(), operatorInputs.getAxes()));
	}

}