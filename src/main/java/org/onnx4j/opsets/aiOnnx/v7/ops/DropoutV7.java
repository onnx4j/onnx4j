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
package org.onnx4j.opsets.aiOnnx.v7.ops;

import java.util.List;

import org.onnx4j.Inputs;
import org.onnx4j.Inputs.Input;
import org.onnx4j.Outputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.model.graph.node.Attributes;
import org.onnx4j.model.graph.node.attributes.FloatAttribute;
import org.onnx4j.opsets.aiOnnx.v6.ops.DropoutV6;

/**
 * Dropout Operator v7
 * 
 * <p>
 * Dropout takes one input data (Tensor) and produces two Tensor outputs, output
 * (Tensor) and mask (Tensor). Depending on whether it is in test mode or not,
 * the output Y will either be a random dropout, or a simple copy of the input.
 * Note that our implementation of Dropout does scaling in the training phase,
 * so during testing nothing needs to be done.
 * 
 * @author HarryLee {@literal <formaten@qq.com>}
 * @version 7
 * @since Version 1 of the default ONNX operator set
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Changelog.md#Dropout-7">
 *      ONNX.Changelog.md</a>
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Operators.md#Dropout">
 *      ONNX.Operators.md</a>
 */
public interface DropoutV7<T_TENSOR> extends DropoutV6<T_TENSOR> {

	public static final String OP_TYPE = "Dropout";

	public static final String ATTR_RATIO = "ratio";

	/**
	 * Executes operator
	 * 
	 * @param data
	 *            The input data as Tensor.
	 * @param ratio
	 *            (float, default 0.5) the ratio of random dropout
	 * @return output : T The output. mask (optional) : T The output mask. If
	 *         is_test is nonzero, this output is not filled.
	 */
	public abstract List<T_TENSOR> dropout(T_TENSOR data, Float ratio);

	@Override
	public default Outputs forward(Node node, Inputs inputs) {
		Attributes attrs = node.getAttrs();

		//
		// (float, default 0.5) the ratio of random dropout
		//
		Float ratio = attrs.getAttrValue(ATTR_RATIO, FloatAttribute.class, 0.5f);

		Input[] inputArray = inputs.get();
		return Outputs.wrap(node, this.dropout(inputArray[0].getTensor(), ratio));
	}

}