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

import java.util.List;

import org.onnx4j.Inputs;
import org.onnx4j.Inputs.Input;
import org.onnx4j.Outputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.model.graph.node.Attributes;
import org.onnx4j.model.graph.node.attributes.IntsAttribute;
import org.onnx4j.opsets.OperatorInputs;
import org.onnx4j.opsets.aiOnnx.v1.AiOnnxOperatorV1;
import org.onnx4j.tensor.DataType;

/**
 * Sigmoid Operator v1
 * 
 * <p>
 * Sigmoid takes one input data (Tensor) and produces one output data (Tensor)
 * where the sigmoid function, y = 1 / (1 + exp(-x)), is applied to the tensor
 * elementwise.
 * 
 * @author HarryLee {@literal <formaten@qq.com>}
 * @version 1
 * @since Version 1 of the default ONNX operator set
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Changelog.md#Sigmoid-1">ONNX.
 *      Changelog.md</a>
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sigmoid">ONNX.
 *      Operators.md</a>
 */
public interface SigmoidV1<T_TENSOR> extends AiOnnxOperatorV1 {

	public static final String OP_TYPE = "Sigmoid";

	class SigmoidInputV1<T_TENSOR> extends OperatorInputs {

		public static final String ATTR_CONSUMED_INPUTS = "consumed_inputs";

		private T_TENSOR x;
		private List<Long> consumedInputs;

		public SigmoidInputV1(Node node, Inputs inputs) {
			super(node, inputs);

			Attributes attrs = node.getAttrs();
			Input[] inputArray = inputs.get();

			//
			// Legacy optimization attribute.
			//
			this.consumedInputs = attrs.getAttrValue(ATTR_CONSUMED_INPUTS, IntsAttribute.class, null);
			this.x = inputArray[0].getTensor();
		}

		public T_TENSOR getX() {
			return x;
		}

		public List<Long> getConsumedInputs() {
			return consumedInputs;
		}

	}

	/**
	 * Executes operator
	 * 
	 * @param x
	 *            Input tensor.
	 * @param consumedInputs
	 *            legacy optimization attribute.
	 * @return Output tensor.
	 */
	public abstract T_TENSOR sigmoid(T_TENSOR x, List<Long> consumedInputs);

	@Override
	public default DataType[] getTypeConstraints() {
		return DataType.floatTypes();
	}

	@Override
	public default OperatorStatus getStatus() {
		return OperatorStatus.STABLE;
	}

	@Override
	public default String getOpType() {
		return OP_TYPE;
	}

	@Override
	public default Outputs forward(Node node, Inputs inputs) {
		SigmoidInputV1<T_TENSOR> operatorInput = new SigmoidInputV1<T_TENSOR>(node, inputs);
		return Outputs.wrap(node, this.sigmoid(operatorInput.getX(), operatorInput.getConsumedInputs()));
	}

}