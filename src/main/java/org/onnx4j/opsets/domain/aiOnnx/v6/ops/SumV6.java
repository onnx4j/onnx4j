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
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.SumV1;
import org.onnx4j.opsets.domain.aiOnnx.v6.AiOnnxOperatorV6;

/**
 * Sum Operator v6
 * 
 * <p>
 * Element-wise sum of each of the input tensors. All inputs and outputs must
 * have the same shape and data type.
 * 
 * @author HarryLee {@literal <formaten@qq.com>}
 * @version 6
 * @since Version 1 of the default ONNX operator set
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Changelog.md#Sum-6">ONNX.
 *      Changelog.md</a>
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sum">ONNX.
 *      Operators.md</a>
 */
public interface SumV6 extends SumV1, AiOnnxOperatorV6 {

	/**
	 * Executes operator
	 * 
	 * @param dataList
	 *            List of tensors for Sum.
	 * @return Output tensor. Same dimension as inputs.
	 */
	//public abstract T_TENSOR sum(List<T_TENSOR> dataList);

	/**
	 * Inputs for operator execution (forward & backward)
	 *
	 * @param <T_TENSOR>
	 *            The backend tensor object.
	 */
	class SumInputsV6<T_TENSOR> extends SumInputsV1<T_TENSOR> {

		public SumInputsV6(Node node, Inputs inputs) {
			super(node, inputs);
		}

		public List<Long> getConsumedInputs() {
			throw new UnsupportedOperationException(
					String.format("Attribute named \"%s\" has deprecated", ATTR_CONSUMED_INPUTS));
		}

	}

	/**
	 * Outputs for operator execution (forward & backward)
	 * 
	 * @param <T_TENSOR>
	 *            The backend tensor object.
	 */
	class SumOutputV6<T_TENSOR> extends SumOutputV1<T_TENSOR> {

		public SumOutputV6(T_TENSOR output) {
			super(output);
		}

	}

}