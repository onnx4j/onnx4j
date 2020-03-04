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

import org.onnx4j.Inputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.model.graph.node.attributes.IntAttribute;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.CastV1;
import org.onnx4j.opsets.domain.aiOnnx.v6.AiOnnxOperatorV6;
import org.onnx4j.opsets.operator.Field;
import org.onnx4j.opsets.operator.Field.TypeConstraint;
import org.onnx4j.opsets.operator.fields.AttributeField;

/**
 * Cast Operator v6
 * 
 * <p>
 * The operator casts the elements of a given input tensor to a data type
 * specified by the 'to' argument and returns an output tensor of the same size
 * in the converted type. The 'to' argument must be one of the data types
 * specified in the 'DataType' enum field in the TensorProto message. NOTE:
 * Casting to and from strings is not supported yet.
 * 
 * @author HarryLee {@literal <formaten@qq.com>}
 * @version 6
 * @since Version 1 of the default ONNX operator set
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Changelog.md#Cast-6">ONNX
 *      .Changelog.md</a>
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Operators.md#Cast">ONNX.
 *      Operators.md</a>
 * @see CastV1
 */
public interface CastV6 extends CastV1, AiOnnxOperatorV6 {

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
	//public T_TENSOR cast(T_TENSOR t1, Long to);

	class CastInputV6<T_TENSOR> extends CastInputsV1<T_TENSOR> {

		private Field<Long> toDTNumberField;

		public CastInputV6(Node node, Inputs inputs) {
			super(node, inputs);

			this.toDTNumberField = new AttributeField<Long>(super.attrs, ATTR_TO, IntAttribute.class, null, true);
		}

		public Long getToDTNumber() {
			return toDTNumberField.getData();
		}
		
	}

	/**
	 * Outputs for operator execution (forward & backward)
	 *
	 * @param <T_TENSOR>
	 *            The backend tensor object.
	 */
	class CastOutputV6<T_TENSOR> extends CastOutputV1<T_TENSOR> {

		public CastOutputV6(T_TENSOR output) {
			super(output);
		}

		@Override
		public TypeConstraint getTypeConstraint() {
			return TPYE_CONSTRAINT_T2;
		}

	}

}