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
package org.onnx4j.opsets.aiOnnx.v8.ops;

import java.util.List;

import org.onnx4j.Inputs;
import org.onnx4j.Outputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.opsets.aiOnnx.v6.ops.SumV6;
import org.onnx4j.opsets.aiOnnx.v8.AiOnnxOperatorV8;

/**
 * Sum Operator v8
 * 
 * <p>
 * Element-wise sum of each of the input tensors (with Numpy-style broadcasting
 * support). All inputs and outputs must have the same data type. This operator
 * supports multidirectional (i.e., Numpy-style) broadcasting; for more details
 * please check the <a href=
 * "https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md">doc</a>.
 * 
 * @author HarryLee {@literal <formaten@qq.com>}
 * @version 8
 * @since Version 1 of the default ONNX operator set
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Changelog.md#Sum-8">ONNX.
 *      Changelog.md</a>
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sum">ONNX.
 *      Operators.md</a>
 */
public interface SumV8<T_TENSOR> extends SumV6<T_TENSOR>, AiOnnxOperatorV8 {

	class SumInputV8<T_TENSOR> extends SumInputV6<T_TENSOR> {

		public SumInputV8(Node node, Inputs inputs) {
			super(node, inputs);
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
		SumInputV8<T_TENSOR> operatorInput = new SumInputV8<T_TENSOR>(node, inputs);
		return Outputs.wrap(node, this.sum(operatorInput.getDataList()));
	}

}