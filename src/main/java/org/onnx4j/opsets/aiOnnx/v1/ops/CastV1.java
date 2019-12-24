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

import org.onnx4j.Inputs;
import org.onnx4j.Inputs.Input;
import org.onnx4j.Outputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.model.graph.node.Attributes;
import org.onnx4j.model.graph.node.attributes.StringAttribute;
import org.onnx4j.opsets.OperatorInputs;
import org.onnx4j.opsets.aiOnnx.v1.AiOnnxOperatorV1;
import org.onnx4j.tensor.DataType;

/**
 * Cast Operator v1
 * 
 * <p>
 * The operator casts the elements of a given input tensor to a data type
 * specified by the 'to' argument and returns an output tensor of the same size
 * in the converted type. The 'to' argument must be one of the data types
 * specified in the 'DataType' enum field in the TensorProto message. NOTE:
 * Casting to and from strings is not supported yet.
 * 
 * @author HarryLee {@literal <formaten@qq.com>}
 * @version 1
 * @since Version 1 of the default ONNX operator set
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Changelog.md#Cast-1">ONNX
 *      .Changelog.md</a>
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Operators.md#Cast">ONNX.
 *      Operators.md</a>
 */
public interface CastV1<T_TENSOR> extends AiOnnxOperatorV1 {

	class CastInputV1<T_TENSOR> extends OperatorInputs {

		public static final String ATTR_TO = "to";

		protected T_TENSOR t1;
		protected String to;

		public CastInputV1(Node node, Inputs inputs) {
			super(node, inputs);

			Attributes attrs = node.getAttrs();
			Input[] inputArray = inputs.get();

			this.t1 = inputArray[0].getTensor();

			this.to = attrs.getAttrValue(ATTR_TO, StringAttribute.class, null);
		}

		public T_TENSOR getT1() {
			return t1;
		}

		public String getTo() {
			return to;
		}
	}

	public static final String OP_TYPE = "Cast";

	/**
	 * Executes operator
	 * 
	 * @param t1
	 *            Input tensor to be cast.
	 * @param to
	 *            The data type to which the elements of the input tensor are
	 *            cast. Strictly must be one of the types from DataType enum in
	 *            TensorProto
	 * @return Output tensor with the same shape as input with type specified by
	 *         the 'to' argument
	 */
	public T_TENSOR cast(T_TENSOR t1, String to);

	@Override
	public default OperatorStatus getStatus() {
		return OperatorStatus.STABLE;
	}

	@Override
	public default String getOpType() {
		return OP_TYPE;
	}

	@Override
	public default DataType[] getTypeConstraints() {
		return DataType.allTypesWithoutStringAndComplex();
	}

	@Override
	public default Outputs forward(Node node, Inputs inputs) {
		CastInputV1<T_TENSOR> input = new CastInputV1<T_TENSOR>(node, inputs);
		return Outputs.wrap(node, this.cast(input.getT1(), input.getTo()));
	}

}