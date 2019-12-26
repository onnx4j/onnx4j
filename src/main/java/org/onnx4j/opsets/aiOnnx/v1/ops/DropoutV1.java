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

import java.util.LinkedList;
import java.util.List;
import java.util.Optional;

import org.onnx4j.Inputs;
import org.onnx4j.Inputs.Input;
import org.onnx4j.Outputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.model.graph.node.Attributes;
import org.onnx4j.model.graph.node.attributes.FloatAttribute;
import org.onnx4j.model.graph.node.attributes.IntAttribute;
import org.onnx4j.model.graph.node.attributes.IntsAttribute;
import org.onnx4j.opsets.aiOnnx.v1.AiOnnxOperatorV1;
import org.onnx4j.tensor.DataType;

/**
 * Dropout Operator v1
 * 
 * <p>
 * Dropout takes one input data (Tensor) and produces two Tensor outputs, output
 * (Tensor) and mask (Tensor). Depending on whether it is in test mode or not,
 * the output Y will either be a random dropout, or a simple copy of the input.
 * Note that our implementation of Dropout does scaling in the training phase,
 * so during testing nothing needs to be done.
 * 
 * @author HarryLee {@literal <formaten@qq.com>}
 * @version 1
 * @since Version 1 of the default ONNX operator set
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Changelog.md#Dropout-1">
 *      ONNX.Changelog.md</a>
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Operators.md#Dropout">
 *      ONNX.Operators.md</a>
 */
public interface DropoutV1<T_TENSOR> extends AiOnnxOperatorV1 {

	public static final String OP_TYPE = "Dropout";

	public static final String ATTR_RATIO = "ratio";

	public static final String ATTR_IS_TEST = "is_test";

	public static final String ATTR_CONSUMED_INPUTS = "consumed_inputs";

	/**
	 * Executes operator
	 * 
	 * @param data
	 *            The input data as Tensor.
	 * @param isTest
	 *            (int, default 0) if nonzero, run dropout in test mode where
	 *            the output is simply Y = X.
	 * @param ratio
	 *            (float, default 0.5) the ratio of random dropout
	 * @param consumedInputs
	 *            legacy optimization attribute.
	 * @return output : T The output. mask (optional) : T The output mask. If
	 *         is_test is nonzero, this output is not filled.
	 */
	public abstract List<T_TENSOR> dropout(T_TENSOR data, Boolean isTest, Float ratio, List<Long> consumedInputs);

	public default List<T_TENSOR> wrapMultiOutputs(Optional<T_TENSOR> data, Optional<T_TENSOR> mask) {
		List<T_TENSOR> outputs = new LinkedList<T_TENSOR>();

		outputs.add(data.orElseThrow(
				() -> new RuntimeException(String.format("[%s] The field named \"data\" can not be null", OP_TYPE))));
		mask.ifPresent((value) -> outputs.add(value));
		return outputs;
	}

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
		// Legacy optimization attribute.
		//
		List<Long> consumedInputs = attrs.getAttrValue(ATTR_CONSUMED_INPUTS, IntsAttribute.class, null);

		//
		// (float, default 0.5) the ratio of random dropout
		//
		Float ratio = attrs.getAttrValue(ATTR_RATIO, FloatAttribute.class, 0.5f);

		//
		// (int, default 0) if nonzero, run dropout in test mode where the
		// output is simply Y = X.
		//
		Boolean isTest = attrs.getAttrValue(ATTR_IS_TEST, IntAttribute.class, 0L).intValue() != 0 ? true : false;

		Input[] inputArray = inputs.get();
		List<T_TENSOR> outputs = this.dropout(inputArray[0].getTensor(), isTest, ratio, consumedInputs);
		return Outputs.wrap(node, outputs);
	}

}