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
package org.onnx4j.tensor;

import org.onnx4j.onnx.prototypes.OnnxProto3.TensorProto;
import org.onnx4j.onnx.prototypes.OnnxProto3.TensorShapeProto;
import org.onnx4j.onnx.prototypes.OnnxProto3.ValueInfoProto;

public class ValueInfo {

	private DataType dataType;
	private Shape shape;

	public static ValueInfo toValueInfo(ValueInfoProto valueInfoProto) {
		TensorProto.DataType dataTypeProto = TensorProto.DataType
				.forNumber(valueInfoProto.getType().getTensorType()
						.getElemType());
		TensorShapeProto shapeProto = valueInfoProto.getType().getTensorType()
				.getShape();
		return new ValueInfo(DataType.from(dataTypeProto),
				Shape.toShape(shapeProto));
	}

	public ValueInfo(DataType dataType, Shape shape) {
		super();
		this.dataType = dataType;
		this.shape = shape;
	}

	public DataType getDataType() {
		return dataType;
	}

	public Shape getShape() {
		return shape;
	}
	
	public int getRank() {
		return this.shape.dims();
	}
	
	@Override
	public String toString() {
		return this.dataType + " -> " + this.shape;
	}

	@Override
	public boolean equals(Object o) {
		if (o != null && o instanceof ValueInfo) {
			return ((ValueInfo) o).getDataType().equals(this.dataType)
					&& ((ValueInfo) o).getShape().equals(this.shape);
		}
		return false;
	}

}