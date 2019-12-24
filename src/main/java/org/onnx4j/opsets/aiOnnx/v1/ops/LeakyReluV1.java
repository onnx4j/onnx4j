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
import org.onnx4j.model.graph.node.attributes.FloatAttribute;
import org.onnx4j.model.graph.node.attributes.IntsAttribute;
import org.onnx4j.opsets.aiOnnx.v1.AiOnnxOperatorV1;
import org.onnx4j.tensor.DataType;

/**
 * LeakyRelu Operator v1
 * 
 * <p>
 * LeakyRelu takes input data (Tensor) and an argument alpha, and produces one
 * output data (Tensor) where the function
 * {@literal f(x) = alpha * x for x < 0, f(x) = x for x >= 0} , is applied to
 * the data tensor elementwise.
 * 
 * @author HarryLee {@literal <formaten@qq.com>}
 * @version 1
 * @since Version 1 of the default ONNX operator set
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Changelog.md#LeakyRelu-1">
 *      ONNX.Changelog.md</a>
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Operators.md#LeakyRelu">
 *      ONNX.Operators.md</a>
 */
public interface LeakyReluV1<T_TENSOR> extends AiOnnxOperatorV1 {

	public static final String OP_TYPE = "LeakyRelu";

	public static final String ATTR_ALPHA = "alpha";

	public static final String ATTR_CONSUMED_INPUTS = "consumed_inputs";

	/**
	 * Executes operator
	 * 
	 * @param x
	 *            Input tensor
	 * @param alpha
	 *            Coefficient of leakage default to 0.01.
	 * @param consumedInputs
	 *            legacy optimization attribute.
	 * @return
	 */
	public abstract T_TENSOR leakyRelu(T_TENSOR x, Float alpha, List<Long> consumedInputs);

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
		Attributes attrs = node.getAttrs();

		//
		// Coefficient of leakage default to 0.01.
		//
		Float alpha = attrs.getAttrValue(ATTR_ALPHA, FloatAttribute.class, 0.01f);

		//
		// Legacy optimization attribute.
		//
		List<Long> consumedInputs = attrs.getAttrValue(ATTR_CONSUMED_INPUTS, IntsAttribute.class, null);

		Input[] inputArray = inputs.get();
		return Outputs.wrap(node, this.leakyRelu(inputArray[0].getTensor(), alpha, consumedInputs));
	}

}