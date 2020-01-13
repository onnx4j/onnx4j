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
import org.onnx4j.Outputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.model.graph.node.Attributes;
import org.onnx4j.model.graph.node.attributes.IntAttribute;
import org.onnx4j.model.graph.node.attributes.IntsAttribute;
import org.onnx4j.opsets.OperatorInputs;
import org.onnx4j.opsets.aiOnnx.v1.AiOnnxOperatorV1;
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
public interface ReduceMaxV1<T_TENSOR> extends AiOnnxOperatorV1 {

	public static final String OP_TYPE = "ReduceMax";

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
	public abstract T_TENSOR reduceMax(T_TENSOR data, List<Long> axes, Long keepdims);

	@Override
	public default DataType[] getTypeConstraints() {
		return DataType.floatTypes();
	}

	@Override
	public default String getOpType() {
		return OP_TYPE;
	}

	class ReduceMaxInputsV1<T_TENSOR> extends OperatorInputs {

		//
		// list of ints
		//
		public static final String ATTR_AXES = "axes";

		//
		// int (default is 1)
		//
		public static final String ATTR_KEEPDIMS = "keepdims";

		protected T_TENSOR data;
		protected List<Long> axes;
		protected Long keepdims;

		public ReduceMaxInputsV1(Node node, Inputs inputs) {
			super(node, inputs);

			Attributes attrs = node.getAttrs();

			this.data = super.getInputArray()[0].getTensor();
			this.axes = attrs.getAttrValue(ATTR_AXES, IntsAttribute.class, Lists.newLinkedList());
			this.keepdims = attrs.getAttrValue(ATTR_KEEPDIMS, IntAttribute.class, 1L);

			for (Long axis : axes) {
				if (axis < 0) {
					throw new IllegalArgumentException(
							String.format("The list of axes%s can not contains negative values.",
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

		public Long getKeepdims() {
			return keepdims;
		}

	}

	@Override
	public default Outputs forward(Node node, Inputs inputs) {
		ReduceMaxInputsV1<T_TENSOR> operatorInputs = new ReduceMaxInputsV1<T_TENSOR>(node, inputs);
		return Outputs.wrap(node,
				this.reduceMax(operatorInputs.getData(), operatorInputs.getAxes(), operatorInputs.getKeepdims()));
	}

}