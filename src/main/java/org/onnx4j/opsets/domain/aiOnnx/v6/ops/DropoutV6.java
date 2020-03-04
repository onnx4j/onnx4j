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
package org.onnx4j.opsets.domain.aiOnnx.v6.ops;

import java.util.List;

import org.onnx4j.Inputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.DropoutV1;
import org.onnx4j.opsets.domain.aiOnnx.v6.AiOnnxOperatorV6;
import org.onnx4j.opsets.operator.OperatorOutputs;

/**
 * Dropout Operator v6
 * 
 * <p>
 * Dropout takes one input data (Tensor) and produces two Tensor outputs, output
 * (Tensor) and mask (Tensor). Depending on whether it is in test mode or not,
 * the output Y will either be a random dropout, or a simple copy of the input.
 * Note that our implementation of Dropout does scaling in the training phase,
 * so during testing nothing needs to be done.
 * 
 * @author HarryLee {@literal <formaten@qq.com>}
 * @version 6
 * @since Version 1 of the default ONNX operator set
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Changelog.md#Dropout-6">
 *      ONNX.Changelog.md</a>
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Operators.md#Dropout">
 *      ONNX.Operators.md</a>
 */
public interface DropoutV6 extends DropoutV1, AiOnnxOperatorV6 {

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
	 * @return output : T The output. mask (optional) : T The output mask. If
	 *         is_test is nonzero, this output is not filled.
	 */
	class DropoutInputsV6<T_TENSOR> extends DropoutInputsV1<T_TENSOR> {

		public DropoutInputsV6(Node node, Inputs inputs) {
			super(node, inputs);
		}

		@Override
		public List<Long> getConsumedInputs() {
			throw new UnsupportedOperationException("Attribute \"consume_inputs\" is not supported in Dropout ver 6");
		}

	}

	/**
	 * Outputs for operator execution (forward & backward)
	 *
	 * @param <T_TENSOR>
	 *            The backend tensor object.
	 */
	class DropoutOutputV6<T_TENSOR> extends DropoutOutputV1<T_TENSOR> {

		public DropoutOutputV6(OperatorOutputs<T_TENSOR> operatorOutputs) {
			super(operatorOutputs);
		}

		public DropoutOutputV6(T_TENSOR data) {
			super(data);
		}

		public DropoutOutputV6(T_TENSOR data, T_TENSOR mask) {
			super(data, mask);
		}

	}

}