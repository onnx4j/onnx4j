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
import org.onnx4j.opsets.aiOnnx.v1.AiOnnxOperatorV1;
import org.onnx4j.tensor.DataType;

/**
 * Relu Operator v1
 * 
 * <p>
 * Relu takes one input data (Tensor) and produces one output data (Tensor)
 * where the rectified linear function, y = max(0, x), is applied to the tensor
 * elementwise.
 * 
 * @author HarryLee {@literal <formaten@qq.com>}
 * @version 1
 * @since Version 1 of the default ONNX operator set
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Changelog.md#Relu-1">ONNX
 *      .Changelog.md</a>
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Operators.md#Relu">ONNX.
 *      Operators.md</a>
 */
public interface ReluV1<T_TENSOR> extends AiOnnxOperatorV1 {

	public static final String OP_TYPE = "Relu";

	public static final String ATTR_CONSUMED_INPUTS = "consumed_inputs";

	/**
	 * Executes operator
	 * 
	 * @param x
	 *            Input tensor
	 * @param consumed_inputs
	 *            legacy optimization attribute
	 * @return Output tensor
	 */
	public abstract T_TENSOR relu(T_TENSOR x, List<Long> consumed_inputs);

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
		// list of ints
		//
		List<Long> consumedInputs = attrs.getAttrValue(ATTR_CONSUMED_INPUTS, IntsAttribute.class, null);

		Input[] inputArray = inputs.get();
		return Outputs.wrap(node, this.relu(inputArray[0].getTensor(), consumedInputs));
	}

}