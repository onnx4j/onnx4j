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

import org.onnx4j.Inputs;
import org.onnx4j.Outputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.opsets.aiOnnx.v6.ops.SubV6;
import org.onnx4j.opsets.aiOnnx.v7.AiOnnxOperatorV7;
import org.onnx4j.tensor.DataType;

/**
 * Sub Operator v7
 * 
 * <p>
 * Performs element-wise binary subtraction (with Numpy-style broadcasting
 * support).
 * 
 * <p>
 * This operator supports multidirectional (i.e., Numpy-style) broadcasting; for
 * more details please check
 * <a href="https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md">the
 * doc</a>.
 * 
 * @author HarryLee {@literal <formaten@qq.com>}
 * @version 7
 * @since Version 1 of the default ONNX operator set
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Changelog.md#Sub-7">ONNX.
 *      Changelog.md</a>
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sub">ONNX.
 *      Operators.md</a>
 */
public interface SubV7<T_TENSOR> extends SubV6<T_TENSOR>, AiOnnxOperatorV7 {

	class SubInputV7<T_TENSOR> extends SubInputV6<T_TENSOR> {

		public SubInputV7(Node node, Inputs inputs) {
			super(node, inputs);
		}

		@Deprecated
		public Long getAxis() {
			throw new UnsupportedOperationException(String.format("Field named \"%s\" has deprecated", ATTR_AXIS));
		}

		@Deprecated
		public Long getBroadcast() {
			throw new UnsupportedOperationException(String.format("Field named \"%s\" has deprecated", ATTR_BROADCAST));
		}

	}

	/**
	 * Executes operator
	 * 
	 * @param a
	 *            First operand.
	 * @param b
	 *            Second operand.
	 * @return Result, has same element type as two inputs
	 */
	public abstract T_TENSOR sub(T_TENSOR a, T_TENSOR b);

	@Override
	public default DataType[] getTypeConstraints() {
		return DataType.floatTypes();
	}

	@Override
	public default OperatorStatus getStatus() {
		return OperatorStatus.STABLE;
	}

	@Override
	public default Outputs forward(Node node, Inputs inputs) {
		SubInputV7<T_TENSOR> operatorInput = new SubInputV7<T_TENSOR>(node, inputs);
		return Outputs.wrap(node, this.sub(operatorInput.getA(), operatorInput.getB()));
	}

}