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
package org.onnx4j.opsets.aiOnnx.v6.ops;

import java.util.List;

import org.onnx4j.Inputs;
import org.onnx4j.Inputs.Input;
import org.onnx4j.Outputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.model.graph.node.Attributes;
import org.onnx4j.model.graph.node.attributes.FloatAttribute;
import org.onnx4j.model.graph.node.attributes.IntAttribute;
import org.onnx4j.opsets.aiOnnx.v1.ops.DropoutV1;

/**
 * Dropout-6
 * 
 * @author HarryLee
 * @see https://github.com/onnx/onnx/blob/master/docs/Changelog.md#Dropout-6
 * @version This version of the operator has been available since version 6 of
 *          the default ONNX operator set.
 *
 */
public interface DropoutV6<T_TENSOR> extends DropoutV1<T_TENSOR> {

	/**
	 * Dropout takes one input data (Tensor) and produces two Tensor outputs,
	 * output (Tensor) and mask (Tensor). Depending on whether it is in test
	 * mode or not, the output Y will either be a random dropout, or a simple
	 * copy of the input. Note that our implementation of Dropout does scaling
	 * in the training phase, so during testing nothing needs to be done.
	 * 
	 * @param data
	 *            The input data as Tensor.
	 * @param isTest
	 *            (int, default 0) if nonzero, run dropout in test mode where
	 *            the output is simply Y = X.
	 * @param ratio
	 *            (float, default 0.5) the ratio of random dropout
	 * @return output : T The output. mask (optional) : T The output mask. If
	 *         is_test is nonzero, this output is not filled.
	 */
	public abstract List<T_TENSOR> dropout(T_TENSOR data, Boolean isTest, Float ratio);

	@Override
	public default Outputs forward(Node node, Inputs inputs) {
		Attributes attrs = node.getAttrs();

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
		return Outputs.wrap(node, this.dropout(inputArray[0].getTensor(), isTest, ratio));
	}

}