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
import org.onnx4j.Outputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.opsets.aiOnnx.v1.ops.SumV1;
import org.onnx4j.opsets.aiOnnx.v6.AiOnnxOperatorV6;

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
public interface SumV6<T_TENSOR> extends SumV1<T_TENSOR>, AiOnnxOperatorV6 {

	class SumInputV6<T_TENSOR> extends SumInputV1<T_TENSOR> {

		public SumInputV6(Node node, Inputs inputs) {
			super(node, inputs);
		}

		@Deprecated
		public List<Long> getConsumedInputs() {
			throw new UnsupportedOperationException(
					String.format("Field named \"%s\" has deprecated", ATTR_CONSUMED_INPUTS));
		}

	}

	/**
	 * Executes operator
	 * 
	 * @param dataList
	 *            List of tensors for Sum.
	 * @return Output tensor. Same dimension as inputs.
	 */
	public abstract T_TENSOR sum(List<T_TENSOR> dataList);

	@Override
	public default OperatorStatus getStatus() {
		return OperatorStatus.STABLE;
	}

	@Override
	public default Outputs forward(Node node, Inputs inputs) {
		SumInputV6<T_TENSOR> operatorInput = new SumInputV6<T_TENSOR>(node, inputs);
		return Outputs.wrap(node, this.sum(operatorInput.getDataList()));
	}

}