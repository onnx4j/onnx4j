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
package org.onnx4j.opsets.domain.aiOnnx.v7.ops;

import org.onnx4j.Inputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.opsets.domain.aiOnnx.v6.ops.DropoutV6;
import org.onnx4j.opsets.domain.aiOnnx.v7.AiOnnxOperatorV7;
import org.onnx4j.opsets.operator.OperatorOutputs;

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
public interface DropoutV7 extends DropoutV6, AiOnnxOperatorV7 {

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
	class DropoutInputsV7<T_TENSOR> extends DropoutInputsV6<T_TENSOR> {
		
		public DropoutInputsV7(Node node, Inputs inputs) {
			super(node, inputs);
		}

		@Override
		public Boolean isTest() {
			throw new UnsupportedOperationException("Attribute \"is_test\" is not supported in Dropout ver 7");
		}

	}

	/**
	 * Outputs for operator execution (forward & backward)
	 *
	 * @param <T_TENSOR>
	 *            The backend tensor object.
	 */
	class DropoutOutputV7<T_TENSOR> extends DropoutOutputV6<T_TENSOR> {

		public DropoutOutputV7(OperatorOutputs<T_TENSOR> operatorOutputs) {
			super(operatorOutputs);
		}

		public DropoutOutputV7(T_TENSOR data) {
			super(data);
		}

		public DropoutOutputV7(T_TENSOR data, T_TENSOR mask) {
			super(data, mask);
		}

	}

}